"""
RAG Retriever for Industrial Predictive Maintenance.
Indexes OEM technical manuals, SOPs, and fault trees using BM25 and TF-IDF scoring.
Runs self-contained and offline without requiring external paid API keys.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import math
import re


class TechnicalManualRetriever:
    def __init__(self, kb_dir: Path = None):
        if kb_dir is None:
            kb_dir = Path(__file__).resolve().parent / "knowledge_base"
        self.kb_dir = kb_dir
        self.documents: List[Dict[str, Any]] = []
        self.corpus: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.load_and_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple, robust alphanumeric tokenizer with lowercase normalization."""
        return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())

    def load_and_index(self):
        """Loads all JSON documents from knowledge_base directory and constructs BM25 index."""
        self.documents = []
        self.corpus = []
        self.df = {}

        if not self.kb_dir.exists():
            return

        json_files = sorted(list(self.kb_dir.glob("*.json")))
        for path in json_files:
            try:
                data = json.loads(path.read_text())
                self.documents.append(data)
                
                # Combine textual fields into indexable text
                text_parts = [
                    data.get("doc_id", ""),
                    data.get("title", ""),
                    data.get("subsystem", ""),
                    " ".join(data.get("affected_sensors", [])),
                    " ".join(str(v) for v in data.get("telemetry_indicators", {}).values()),
                    " ".join(data.get("root_causes", [])),
                    data.get("dispatch_protocol", ""),
                    " ".join(data.get("inspection_procedure", [])),
                    data.get("safety_advisories", "")
                ]
                tokens = self._tokenize(" ".join(text_parts))
                self.corpus.append(tokens)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        # Compute document lengths & averages
        self.doc_lengths = [len(doc) for doc in self.corpus]
        N = len(self.documents)
        self.avg_doc_len = (sum(self.doc_lengths) / N) if N > 0 else 0.0

        # Compute document frequencies (DF)
        for doc in self.corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1

        # Compute Robertson-Spärck Jones IDF
        for term, freq in self.df.items():
            self.idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)

    def retrieve(self, query: str, top_k: int = 3, k1: float = 1.5, b: float = 0.75) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant documents using BM25 ranking algorithm.
        """
        if not self.documents:
            return []

        q_tokens = self._tokenize(query)
        scores = [0.0] * len(self.documents)

        for i, doc_tokens in enumerate(self.corpus):
            doc_len = self.doc_lengths[i]
            # term frequencies in this document
            tf: Dict[str, int] = {}
            for t in doc_tokens:
                tf[t] = tf.get(t, 0) + 1

            doc_score = 0.0
            for qt in q_tokens:
                if qt in tf:
                    freq = tf[qt]
                    idf_val = self.idf.get(qt, 0.1)
                    # BM25 numerator and denominator
                    num = freq * (k1 + 1.0)
                    denom = freq + k1 * (1.0 - b + b * (doc_len / (self.avg_doc_len or 1.0)))
                    doc_score += idf_val * (num / denom)

            scores[i] = doc_score

        # Rank documents by score
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

        results = []
        for idx in ranked_indices[:top_k]:
            results.append({
                "document": self.documents[idx],
                "score": round(scores[idx], 3)
            })

        return results

    def retrieve_by_sensors(self, sensors: List[str], top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieves documents specifically targeting an anomalous set of sensors."""
        query_text = " ".join(sensors)
        return self.retrieve(query_text, top_k=top_k)
