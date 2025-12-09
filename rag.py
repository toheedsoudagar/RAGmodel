# rag.py
import os
import re
from collections import defaultdict
import numpy as np

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingest import run_ingestion_if_needed
from sql_agent import SQLAgent
from db_setup import create_database_from_sql_files, DB_URI

# ---------- Configuration ----------
EMBEDDING_MODEL = "embeddinggemma:latest"
CHROMA_DB_DIR = "chroma_db"
SCORE_THRESHOLD = 0.1  # Stabilized RAG threshold
MAX_PER_SOURCE = 6
CANDIDATE_POOL = 20
FINAL_K = 6
LLM_MODEL = "gemma3:1b"
LLM_TEMPERATURE = 0.1
SQL_DB_URI = DB_URI
# -----------------------------------

class RAGPipeline:
    def __init__(self):
        # 1. SETUP: Create the live SQL database schema from .sql files
        print("[INIT] Setting up SQL database...")
        self.live_db_uri = create_database_from_sql_files(SQL_DB_URI)

        # 2. RAG INGESTION: Ensure document ingestion is run
        run_ingestion_if_needed()

        # RAG Components
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # 3. Load the Chroma DB
        print("[INIT] Loading persisted Chroma DB...")
        self.db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embeddings
        )

        # LLM (Used by both RAG and the SQL agent internally)
        self.llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

        # 4. SQL Agent Component
        self.sql_agent = SQLAgent(
            db_uri=self.live_db_uri or "",
            llm_model=LLM_MODEL,
            llm_temperature=LLM_TEMPERATURE
        )

        self.chat_history = []

    # --- RAG Helper Methods ---
    @staticmethod
    def _cosine(v1, v2):
        v1 = np.array(v1, dtype=float)
        v2 = np.array(v2, dtype=float)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0: return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    @staticmethod
    def _compact_text(text, max_chars=1500):
        if not text: return ""
        t = re.sub(r'\s+', ' ', text).strip()
        return t[:max_chars]

    @staticmethod
    def _dedupe_sentences(text, max_output_chars=1200):
        if not text: return ""
        parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
        seen = set()
        out = []
        for p in parts:
            s = p.strip()
            if not s: continue
            key = s.lower()
            if key in seen: continue
            seen.add(key)
            out.append(s)
            if sum(len(x) for x in out) > max_output_chars: break
        return " ".join(out)

    def retrieve(self, query, k=FINAL_K, debug=True, force_source=None, source_boost=None,
                 score_threshold=SCORE_THRESHOLD, max_per_source=MAX_PER_SOURCE,
                 candidate_pool=CANDIDATE_POOL):

        if debug: print(f"[retrieve] Query={query!r} k={k} force_source={force_source} score_threshold={score_threshold}")

        # 1. Embed the query
        q_emb = self.embeddings.embed_query(query)

        # 2. Fetch ALL documents with embeddings for manual scoring
        # (Standard similarity_search returns docs WITHOUT embeddings, causing your logic to fail)
        candidates = []
        try:
            if debug: print(f"[retrieve] Fetching all docs via db.get() for manual scoring...")
            
            # Fetch content, embeddings, and metadata
            res = self.db.get(include=["documents", "embeddings", "metadatas"])
            docs_list = res.get("documents", [])
            embs_list = res.get("embeddings", [])
            metas_list = res.get("metas", []) if res.get("metas") else res.get("metadatas", [])

            # Reconstruct into temporary objects
            for doc_content, emb_data, meta_data in zip(docs_list, embs_list, metas_list):
                class _D: pass
                d = _D()
                d.page_content = doc_content
                d.metadata = meta_data if meta_data else {}
                d._embedding = emb_data  # Manually attach embedding
                candidates.append(d)

        except Exception as e:
            print(f"[retrieve] Error fetching documents: {e}")
            return []

        # --- 3. Extract and Score Candidates ---
        texts, embeddings, metas, extracted = [], [], [], 0
        for c in candidates:
            emb = getattr(c, "_embedding", None)
            if emb is not None:
                embeddings.append(emb)
                texts.append(getattr(c, "page_content", "") or "")
                metas.append(getattr(c, "metadata", {}) or {})
                extracted += 1

        if extracted == 0:
            if debug: print("[retrieve] No docs/embs available after candidate filtering.")
            return []

        scored = []
        for idx in range(len(texts)):
            meta = metas[idx] if metas else {}
            src = meta.get("source", "unknown") if isinstance(meta, dict) else "unknown"

            # Filtering and Boosting Logic
            if force_source and (force_source.lower() not in src.lower()): continue

            base_score = self._cosine(q_emb, embeddings[idx])
            boost = 1.0
            if source_boost and isinstance(source_boost, dict):
                for key, factor in source_boost.items():
                    if key.lower() in src.lower():
                        try: boost = float(factor)
                        except Exception: boost = 1.0
                        break
            final_score = base_score * boost
            scored.append({"score": final_score, "text": texts[idx], "meta": meta, "src": src, "idx": idx})

        # --- 4. Final Selection ---
        scored.sort(key=lambda x: x["score"], reverse=True)
        filtered = [s for s in scored] # Optional: if s["score"] >= score_threshold

        if debug: print(f"[retrieve] scored: {len(scored)} ; after threshold: {len(filtered)}")

        by_source_count = defaultdict(int)
        final_items = []
        for item in filtered:
            if by_source_count[item["src"]] >= max_per_source: continue
            by_source_count[item["src"]] += 1
            final_items.append(item)
            if len(final_items) >= k: break

        if debug:
            print("[retrieve] Final items:")
            for i, it in enumerate(final_items, 1):
                preview = (it["text"] or "")[:200].replace("\n", " ")
                print(f" {i:02d}. score={it['score']:.4f} src={it['src']} preview={preview}")

        docs_out = [Document(page_content=it["text"], metadata=it["meta"] or {}) for it in final_items]
        return docs_out

    def format_context(self, docs):
        if not docs: return ""
        grouped = defaultdict(list)
        for d in docs:
            src = (d.metadata.get("source") if isinstance(d.metadata, dict) else "") or "unknown"
            compacted = self._compact_text(d.page_content, max_chars=2000)
            grouped[src].append(compacted)

        sections = []
        for src, pieces in grouped.items():
            merged = " ".join(pieces)
            deduped = self._dedupe_sentences(merged)
            sections.append(f"[{src}]\n{deduped}\n")
        return "\n\n".join(sections)

    # --- Query Router ---
    def _is_sql_query(self, query: str) -> bool:
        """Simple keyword heuristic to route the query."""
        keywords = [
            "total", "sum", "average", "count", "latest", "oldest", "max", "min",
            "highest", "lowest", "calculate", "inventory", "sales", "revenue",
            "price", "list", "table", "database", "schema", "show", "columns"
        ]
        q = query.lower()
        if any(kw in q for kw in keywords):
            return True
        return False

    # --- Main Ask Method ---
    def ask(self, query):
        if self._is_sql_query(query):
            print(f"[Router] Routing query to SQL Agent: {query!r}")
            summary, raw_table = self.sql_agent.ask(query)
            final_answer = f"{summary}\n\n**Raw Query Results:**\n```\n{raw_table}\n```"
            self.chat_history.append({"user": query, "assistant": final_answer})
            return final_answer, []

        else:
            print(f"[Router] Routing query to Document RAG: {query!r}")
            docs = self.retrieve(query, k=FINAL_K, debug=True)
            context = self.format_context(docs)

            if len(context.strip()) == 0:
                answer = "I don't have enough information in the documents to answer that question."
                return answer, docs

            messages = [
                {"role": "system", "content":
                    "You are a RAG assistant. Use ONLY the provided context. "
                    "When analyzing data from CSV or Excel files, recognize that the context may be presented in a structured key:value format. "
                    "Write a **detailed and comprehensive answer** in clear, natural paragraphs. "
                    "Do NOT copy verbatim; synthesize information. "
                    "If a fact comes from a document, include its source in brackets (e.g. [file.pdf])."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]

            response = self.llm.invoke(messages)
            raw_answer = response.content
            
            # Simple deduplication for answer lines
            lines = [ln.strip() for ln in raw_answer.splitlines() if ln.strip()]
            out_lines = []
            seen = set()
            for ln in lines:
                if ln.lower() in seen: continue
                seen.add(ln.lower())
                out_lines.append(ln)
            final_answer = "\n".join(out_lines)

            self.chat_history.append({"user": query, "assistant": final_answer})
            return final_answer, docs