import json
import numpy as np
import faiss

import sys
sys.path.append("C:\\Users\\matse\\gig\\src\\system_v5") # Adjust this path to point to the root of your project if needed

from sentence_transformers import SentenceTransformer
from backends import load_backend


class NaiveRAG:
    def __init__(
        self,
        chunk_path=r"C:\Users\matse\gig\src\system_v5\content\chunks.jsonl",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
    ):
        # 1. Load local phi-4-mini-instruct backend
        self.backend = load_backend(name="phi-npu-openvino")

        # 2. Load pre-chunked documents from JSONL
        self.documents = self._load_chunks_from_jsonl(chunk_path)

        # 3. Load embedding model
        self.embedder = SentenceTransformer(embedding_model_name)

        # 4. Embed all documents
        self.doc_embeddings = self.embedder.encode(
            self.documents,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        # 5. Build FAISS index
        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product (cosine if normalized)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.doc_embeddings)
        self.index.add(self.doc_embeddings)

        # 6. Retrieval config
        self.top_k = top_k

    # ----------------- helpers -----------------

    def _load_chunks_from_jsonl(self, path: str):
        documents = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("chunk_text_clean", "").strip()
                if text:
                    documents.append(text)
        return documents

    def _retrieve(self, query: str):
        query_emb = self.embedder.encode(
            [query],
            convert_to_numpy=True,
        )
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, self.top_k)
        indices = indices[0]
        scores = scores[0]
        retrieved = [(self.documents[i], float(scores[idx])) for idx, i in enumerate(indices)]
        return retrieved

    def _build_prompt(self, question: str, contexts: list[str]) -> str:
        context_block = "\n\n---\n\n".join(contexts)
        prompt = f"""You are a helpful assistant.
        Use the provided context to answer the question."

        Context:
        {context_block}

        Question:
        {question}

        Answer:"""
        return prompt

    def _generate(self, prompt: str) -> str:
        return self.backend.generate(
            prompt,
            max_new_tokens=256,
            json_schema=None,
        )

    # ----------------- public API -----------------

    def run(self, question: str) -> dict:
        retrieved = self._retrieve(question)
        contexts = [c for c, _ in retrieved]
        prompt = self._build_prompt(question, contexts)
        answer = self._generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "scores": [s for _, s in retrieved],
        }

if __name__ == "__main__":
    rag = NaiveRAG()
    question = input("Enter your question: ")
    result = rag.run(question)
    answer_text = result["answer"]
    print(json.dumps(result, indent=2))
