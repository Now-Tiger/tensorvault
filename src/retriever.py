#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# retriever.py
# The Search & Rerank Pipeline
# This is where the magic happens. We retrieve top-K candidates using fast vector search, then use a Cross-Encoder to rigorously evaluate and re-rank the text-based results against the query
from __future__ import annotations

import logging
from typing import List

from sentence_transformers import CrossEncoder

from src.embedder import MultimodalEmbedder
from src.vault_db import VaultDatabase
from src.types_cofig import Modality, RetrievalResult, VaultAsset

# iniit logger
logger = logging.getLogger(__name__)


class HybridRetriever:

    def __init__(self, embedder: MultimodalEmbedder, db: VaultDatabase):
        self.embedder = embedder
        self.db = db
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def query_vault(self, query: str, creator_id: str, top_k: int = 5) -> List[RetrievalResult]:
        logger.info(f"Executing retrieval for query: '{query}'")

        # 1. Embed search query
        query_vector = self.embedder.embed_text([query])[0]

        # 2. First Stage: Dense Retrieval 
        # (Fetch more candidates than top_k to account for duplicates we will drop later)
        candidates = self.db.search_assets(query_vector=query_vector, creator_id=creator_id, top_k=20)

        if not candidates:
            return []

        # 3. Second Stage: Re-ranking
        reranked_results = []
        text_candidates = []
        visual_results = [] 

        for item in candidates:
            asset = VaultAsset(
                id=str(item.id),
                creator_id=item.payload["creator_id"],
                modality=item.payload["modality"],
                content_uri=item.payload["content_uri"],
                caption=item.payload.get("caption"),
                metadata=item.payload.get("metadata", {}) 
            )

            if asset.modality == Modality.TEXT or asset.modality == "text":
                text_candidates.append(asset)
            else:
                visual_results.append(RetrievalResult(asset=asset, score=item.score))

        if text_candidates:
            pairs = [[query, doc.content_uri] for doc in text_candidates]
            cross_scores = self.reranker.predict(pairs)

            for asset, score in zip(text_candidates, cross_scores):
                reranked_results.append(RetrievalResult(asset=asset, score=float(score)))

        # Combine and sort all results highest to lowest
        final_results = reranked_results + visual_results
        final_results.sort(key=lambda x: x.score, reverse=True)

        # 4. Final Stage: Deduplication (Group by source file)
        deduped_results = []
        seen_uris = set()

        for res in final_results:
            uri = res.asset.content_uri
            
            # If we haven't seen this source file (video or image) yet, add it.
            # Because the list is sorted by score, this guarantees we keep 
            # only the single highest-scoring frame for any given video.
            if uri not in seen_uris:
                seen_uris.add(uri)
                deduped_results.append(res)
                
            # Stop once we have enough unique files to satisfy top_k
            if len(deduped_results) >= top_k:
                break

        return deduped_results
