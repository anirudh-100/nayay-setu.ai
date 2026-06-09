"""NyaySetu RAG engine: hybrid retrieval, reranking, and citation-grounded answers.

This package replaces the two earlier, divergent implementations (the in-memory
``faiss_search`` path and the orphaned ``app/services`` FAISS path) with a single
persisted, metadata-aware pipeline.
"""
