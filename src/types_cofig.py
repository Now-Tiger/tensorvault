#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# types_config.py (Strict Data Contracts)
# Using pydantic ensures our data structures are rigorously validated before they ever touch the embedding models or the database.
from __future__ import annotations

from enum import Enum
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field


class Modality(str, Enum):

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class VaultAsset(BaseModel):

    id: str
    creator_id: str
    modality: Modality
    content_uri: str  # Path to image/video or actual text content
    caption: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):

    asset: VaultAsset
    score: float
