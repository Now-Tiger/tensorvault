#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
import shutil
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.embedder import MultimodalEmbedder
from src.retriever import HybridRetriever
from src.types_cofig import Modality, VaultAsset
from src.vault_db import VaultDatabase
from src.video_frames_extractor import extract_video_frames

logger = logging.getLogger(__name__)
_ = load_dotenv(find_dotenv('.env'))


# Global dictionary to hold our heavy ML singletons
services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ML Models into memory... This may take a minute.")
    embedder = MultimodalEmbedder()
    db = VaultDatabase()
    retriever = HybridRetriever(embedder=embedder, db=db)

    services["embedder"] = embedder
    services["db"] = db
    services["retriever"] = retriever
    print("API is ready to receive traffic!")
    yield
    # Cleanup resources on shutdown
    services.clear()


# Init fastapi app
app = FastAPI(title="Multimodal Vault API", lifespan=lifespan)


# Mount local storage replace later with object storage such as S3
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# Helper dependency to inject services into routes
def get_retriever() -> HybridRetriever:
    return services["retriever"]


def get_embedder() -> MultimodalEmbedder:
    return services["embedder"]


def get_db() -> VaultDatabase:
    return services["db"]


@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


@app.post("/upload/image/")
async def upload_image(creator_id: str = Form(...), caption: Optional[str] = Form(None), file: UploadFile = File(...), embedder: MultimodalEmbedder = Depends(get_embedder), db: VaultDatabase = Depends(get_db)):
    """Uploads an image (with optional caption), generates a CLIP embedding, and stores it in Qdrant."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    _ = os.makedirs("uploads/images", exist_ok=True)
    file_path = f"uploads/images/{file.filename}"

    with open(file_path, "wb") as buffer:
        _ = shutil.copyfileobj(file.file, buffer)

    try:
        # Note: You will need to update your VaultAsset type to accept 'caption'
        asset = VaultAsset(
            id=file.filename, 
            creator_id=creator_id,
            modality=Modality.IMAGE,
            content_uri=file_path,
            caption=caption # Store the caption in the asset
        )

        # We are still only embedding the IMAGE visually
        vectors = embedder.embed_images([file_path])

        # When db.upsert_assets runs, it should be updated to push asset.caption into the Qdrant payload
        _ = db.upsert_assets([asset], vectors)

        return {
            "status": "success", 
            "message": "Image embedded and stored.", 
            "asset_id": asset.id,
            "caption_saved": bool(caption)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/video/")
async def upload_video(creator_id: str = Form(...), caption: Optional[str] = Form(None), file: UploadFile = File(...), embedder: MultimodalEmbedder = Depends(get_embedder), db: VaultDatabase = Depends(get_db)):
    """Uploads a video, processes frame samples at checkpoints, and stores them in Qdrant."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded asset must be a valid video container.")

    # 1. Persistent directories setup
    _ = os.makedirs("uploads/videos", exist_ok=True)
    base_name  = os.path.splitext(file.filename)[0]
    frames_dir = f"uploads/frames/{base_name}"
    video_path = f"uploads/videos/{file.filename}"

    # 2. Save parent file to disk
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 3. Process video into individual frames
        frames_meta = extract_video_frames(video_path, frames_dir, frame_rate=1)
        
        assets = []
        image_paths_to_embed = []
        
        # 4. Loop over milestones to generate unique Qdrant targets
        for _, item in enumerate(frames_meta):
            frame_path = item["frame_path"]
            timestamp = item["timestamp"]
            
            # Create a unique point string ID combining video filename + exact frame timestamp
            unique_frame_id = f"{file.filename}_{timestamp}"
            
            asset = VaultAsset(
                id=unique_frame_id,
                creator_id=creator_id,
                modality=Modality.VIDEO,
                content_uri=video_path, # Main URI points directly to video file
                caption=caption,
                metadata={
                    "timestamp": timestamp,
                    "thumbnail_uri": frame_path # Store specific matching frame image path
                }
            )

            _ = assets.append(asset)
            _ = image_paths_to_embed.append(frame_path)

        if not assets:
            raise HTTPException(status_code=400, detail="Video file too short or formatting corrupted.")

        # 5. Bulk embed all frame captures at once via CLIP
        vectors = embedder.embed_images(image_paths_to_embed)

        # 6. Multi-point index upsert to Qdrant
        _ = db.upsert_assets(assets, vectors)

        return {
            "status": "success",
            "message": f"Successfully parsed video. Indexed {len(assets)} vector windows.",
            "video_path": video_path
        }

    except Exception as e:
        # Fallback cleanup
        if os.path.exists(video_path):
            _ = os.remove(video_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/", response_model=List[Dict[str, Any]])
async def query_vault(query: str, creator_id: str, top_k: int = 5, retriever: HybridRetriever = Depends(get_retriever)):
    """Searches the creator's vault using text querying against image/text embeddings."""
    try:
        # Request a few extra items in case deduplication shrinks the pool
        results = retriever.query_vault(query=query, creator_id=creator_id, top_k=top_k)

        # Serialize Pydantic objects for JSON response
        return [
            {
                "score": round(res.score, 4),
                "modality": res.asset.modality.value,
                "content_uri": res.asset.content_uri,
                "caption": res.asset.caption,
                "id": res.asset.id,
                "metadata": res.asset.metadata
            }
            for res in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
