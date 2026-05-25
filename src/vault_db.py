#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vault_db.py
# Vector Storage & Retrieval
# This acts as the interface to Qdrant. In production, you would configure this to connect to a remote Qdrant cluster, but we'll use local persistence here.
from __future__ import annotations

import os
import uuid
from typing import Any, List

from dotenv import find_dotenv, load_dotenv
from qdrant_client import QdrantClient, models

# load env
_ = load_dotenv(find_dotenv())


class VaultDatabase:

    def __init__(self, collection_name: str = "creator_vault", vector_size: int = 512):
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))

        if host:
            self.client = QdrantClient(host=host, port=port)
            print(f"Connected to Qdrant at {host}:{port}")
        else:
            self.client = QdrantClient(path="./qdrant_storage")
            print("Connected to local Qdrant instance")

        self.collection_name = collection_name

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert_assets(self, assets: List[Any], vectors: List[List[float]]):
        """Inserts or updates assets and their embeddings in Qdrant."""
        points = []

        for asset, vector in zip(assets, vectors):
            try:
                valid_id = str(uuid.UUID(str(asset.id)))
            except ValueError:
                valid_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(asset.id)))

            points.append(
                models.PointStruct(
                    id=valid_id,
                    vector=vector,
                    payload={
                        "original_id": asset.id,
                        "creator_id": asset.creator_id,
                        "modality": asset.modality.value,
                        "content_uri": asset.content_uri,
                        "caption": asset.caption,
                        "metadata": asset.metadata,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search_assets(self, query_vector: List[float], creator_id: str, top_k: int = 5) -> List[Any]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="creator_id", match=models.MatchValue(value=creator_id)
                    )
                ]
            ),
            limit=top_k,
            with_payload=True,
        )
        return results.points
