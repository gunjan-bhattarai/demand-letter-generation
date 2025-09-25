# demand-letter-generation
Demand letter generation using MCP servers and gemini-cli (with RAG implementation as well).

EMBEDDING_PROVIDER
- Use EMBEDDING_PROVIDER="huggingface" if you want to use specialized embeddings
- Use EMBEDDING_PROVIDER="together" if you want to use standard open-source (but optimized for inference) embeddings with Together.ai
- Use EMBEDDING_PROVIDER="together" if you want to use standard Gemini embeddings

From my review, the differences between the two embedding models are subtle but meaningful in specific contexts. legal-bge-m3 ("huggingface") excels in retrieving nuanced legal and medical details, while the other two provides slightly more specific factual details in some cases ("together").

