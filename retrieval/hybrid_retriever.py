import re
import pickle
from pathlib import Path
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
BM25_INDEX_PATH = CHROMA_DIR / "bm25_index.pkl"
COLLECTION_NAME = "gsstb_textbooks"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DENSE_TOP_K = 25
SPARSE_TOP_K = 25
RRF_K = 60
RERANK_TOP_N = 8

@dataclass
class Retrieved:
    
    chunk_id: str
    text: str
    textbook_name: str
    standard: str
    subject: str
    page_number: int
    score: float

class HybridRetriever:
    def __init__(self):
        self._client=chromadb.PersistentClient(path=str(CHROMA_DIR))
        embed_fn=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
        self._collection=self._client.get_collection(name=COLLECTION_NAME,embedding_function=embed_fn)

        with open(BM25_INDEX_PATH,"rb") as f:
            bm25_data=pickle.load(f)
        self._bm25 = bm25_data["bm25"]
        self._bm25_chunk_ids = bm25_data["chunk_ids"]
        self._chunk_meta = bm25_data["chunk_meta"]
        self._reranker = CrossEncoder(RERANKER_MODEL_NAME)
    
    def _build_chroma_where(self, where: dict | None) -> dict | None:
        if not where:
            return None
        if len(where) == 1:
            key, value = next(iter(where.items()))
            return {key: value}
        return {"$and": [{k: v} for k, v in where.items()]}

    def _dense_search(self, query: str, where: dict | None)->list[tuple[str,int]]:
        result=self._collection.query(
            query_texts=[query],
            n_results=DENSE_TOP_K,
            where=self._build_chroma_where(where)
        )
        chunk_ids=result["ids"][0] if result["ids"] else []
        return [(cid, rank) for rank, cid in enumerate(chunk_ids)]

    def _sparse_search(self, query: str, where: dict | None)->list[tuple[str,int]]: # ranks the chunk_ids based on sparse retrieval scores
        tokenized_query = re.findall(r"\b\w+\b", query.lower())
        scores = self._bm25.get_scores(tokenized_query)

        scored = list(zip(self._bm25_chunk_ids, scores))

        if where:
            def matches(cid):
                meta = self._chunk_meta[cid]
                return all(meta.get(k) == v for k, v in where.items())
            scored = [(cid, s) for cid, s in scored if matches(cid)]

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:SPARSE_TOP_K]
        return [(cid, rank) for rank, (cid, _) in enumerate(top)]
        
    def _reciprocal_rank_fusion(self, dense: list[tuple[str, int]], sparse: list[tuple[str, int]]) -> list[str]:
            scores: dict[str, float] = {} # dictionary to store the fused scores for each chunk_id
            for cid, rank in dense: # iterate through the dense results
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)# add the dense score to the existing score if the chunk_id is already in the scores dictionary, otherwise initialize it with the dense score
            for cid, rank in sparse:
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1) # add the sparse score to the existing score if the chunk_id is already in the scores dictionary, otherwise initialize it with the sparse score

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True) # sort by score, descending
            return [cid for cid, _ in ranked] # only get the names of the chunks, not the scores
    
    def retrieve(self, query: str, standard: str | None = None, subject: str | None = None) -> list[Retrieved]:
        where = {}
        if standard:
            where["standard"] = standard
        if subject:
            where["subject"] = subject
        where = where or None

        dense_ranked = self._dense_search(query, where)
        sparse_ranked = self._sparse_search(query, where)
        fused_ids = self._reciprocal_rank_fusion(dense_ranked, sparse_ranked)

        if not fused_ids:
            return []

        candidate_texts = [self._chunk_meta[cid]["text"] for cid in fused_ids]
        pairs = [[query, text] for text in candidate_texts]
        rerank_scores = self._reranker.predict(pairs)

        reranked = sorted(zip(fused_ids, rerank_scores), key=lambda x: x[1], reverse=True)
        top_chunks = reranked[:RERANK_TOP_N]

        results = []
        for cid, score in top_chunks:
            meta = self._chunk_meta[cid]
            results.append(Retrieved(
                chunk_id=cid,
                text=meta["text"],
                textbook_name=meta["textbook_name"],
                standard=meta["standard"],
                subject=meta["subject"],
                page_number=meta["page_num"],
                score=float(score),
            ))
        return results

