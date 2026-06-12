import numpy as np
from rank_bm25 import BM25Okapi
import faiss
from loguru import logger
from typing import List, Dict

class HybridRetriever:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.bm25 = None
        self.faiss_index = None
        self.candidate_ids = []
        
    def fit(self, candidate_ids: List[str], documents: List[str]):
        """
        Builds both the BM25 and FAISS indices.
        """
        logger.info(f"Building Hybrid Index for {len(documents)} documents...")
        self.candidate_ids = candidate_ids
        
        # Build BM25
        logger.info("Tokenizing for BM25...")
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Build FAISS
        logger.info("Generating dense embeddings...")
        embeddings = self.embedding_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        d = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(embeddings)
        logger.info("Hybrid Index built successfully.")
        
    def retrieve(self, query: str, negative_query: str = None, top_k: int = 500) -> Dict[str, float]:
        """
        Returns a dict of candidate_id -> hybrid_score using RRF.
        Incorporates Negative Vector Search if negative_query is provided.
        """
        if self.bm25 is None or self.faiss_index is None:
            raise ValueError("Must call fit() before retrieve().")
            
        logger.info(f"Retrieving top {top_k} for query: {query}")
        if negative_query:
            logger.info(f"Applying Negative Semantic Anchor: {negative_query[:100]}...")
        
        # Sparse Retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        
        # Dense Retrieval (FAISS) Positive
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        # We need the actual similarity scores for all candidates to apply negative penalty
        # faiss_index.search returns top k, so we search all
        dense_scores_pos, dense_indices_pos = self.faiss_index.search(query_emb, len(self.candidate_ids))
        dense_scores_pos = dense_scores_pos[0]
        dense_indices_pos = dense_indices_pos[0]
        
        # Map original indices to their positive similarity score
        idx_to_pos_score = {idx: score for idx, score in zip(dense_indices_pos, dense_scores_pos)}
        
        # Dense Retrieval (FAISS) Negative
        idx_to_neg_score = {}
        if negative_query:
            neg_emb = self.embedding_model.encode([negative_query], convert_to_numpy=True)
            faiss.normalize_L2(neg_emb)
            dense_scores_neg, dense_indices_neg = self.faiss_index.search(neg_emb, len(self.candidate_ids))
            idx_to_neg_score = {idx: score for idx, score in zip(dense_indices_neg[0], dense_scores_neg[0])}
            
        # Compute adjusted dense scores
        adjusted_dense_scores = np.zeros(len(self.candidate_ids))
        for i in range(len(self.candidate_ids)):
            pos_score = idx_to_pos_score.get(i, 0.0)
            neg_score = idx_to_neg_score.get(i, 0.0)
            # Penalize candidates who are too close to the negative anchor
            adjusted_dense_scores[i] = pos_score - (neg_score * 0.5)
            
        # Re-rank dense based on adjusted scores
        dense_ranks = np.argsort(adjusted_dense_scores)[::-1]
        
        # RRF (Reciprocal Rank Fusion)
        rrf_scores = {}
        k_rrf = 60
        
        # Map indices to ranks
        bm25_rank_dict = {idx: rank for rank, idx in enumerate(bm25_ranks)}
        dense_rank_dict = {idx: rank for rank, idx in enumerate(dense_ranks)}
        
        for i, cid in enumerate(self.candidate_ids):
            rank_sparse = bm25_rank_dict.get(i, len(self.candidate_ids))
            rank_dense = dense_rank_dict.get(i, len(self.candidate_ids))
            
            # Hybrid score formulation
            score = (1.0 / (k_rrf + rank_sparse)) + (1.0 / (k_rrf + rank_dense))
            rrf_scores[cid] = score
            
        # Sort and return top_k
        sorted_candidates = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Normalize scores to 0-1 for S1 component downstream
        top_cands = sorted_candidates[:top_k]
        
        if not top_cands:
            return {}
            
        max_score = top_cands[0][1]
        if max_score == 0:
            return {cid: 0.0 for cid, score in top_cands}
            
        return {cid: (score / max_score) for cid, score in top_cands}
