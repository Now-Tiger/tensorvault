#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# embedder.py
# The Multimodal Engine
# This handles the heavy lifting of mapping content to the shared latent space. It dynamically utilizes GPU/Apple Silicon if available.
from __future__ import annotations

import logging
import os
from typing import List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class MultimodalEmbedder:

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Loading CLIP model on {self.device}...")

        self.hf_token = os.environ.get('HF_TOKEN', None)
        self.model = CLIPModel.from_pretrained(model_name, token=self.hf_token).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, token=self.hf_token)
        self.vector_size = self.model.config.projection_dim

    @torch.no_grad()
    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Embeds text into the shared latent space."""
        inputs = (
            self.processor(
                text=texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            .to(self.device)
        )
        
        outputs = self.model.get_text_features(**inputs)
        
        if not isinstance(outputs, torch.Tensor):
            if hasattr(outputs, "text_embeds"):
                embeddings = outputs.text_embeds
            elif hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs[0]  # Ultimate fallback
        else:
            embeddings = outputs

        # L2 Normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True) 
        return embeddings.cpu().tolist()

    @torch.no_grad()
    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        """Embeds images into the shared latent space."""
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        
        outputs = self.model.get_image_features(**inputs)
        
        if not isinstance(outputs, torch.Tensor):
            if hasattr(outputs, "image_embeds"):
                embeddings = outputs.image_embeds
            elif hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs[0]  # Ultimate fallback
        else:
            embeddings = outputs

        # L2 Normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True) 
        return embeddings.cpu().tolist()
