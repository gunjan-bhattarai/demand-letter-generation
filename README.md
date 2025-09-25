# demand-letter-generation
Demand letter generation using MCP servers and gemini-cli (with RAG implementation as well).

Two MCP servers were ultimately created - one for querying the database for case details, plaintiff info, defendant info, etc. and the other to conduct traditional retrieval augmented generation (RAG). I ended up converting my RAG code into an MCP server in order to allow gemini-cli (my MCP client of choice for this project) to access it. However, Gemini repeatedly preferred to use its own ReadFile tool in order to acquire the needed information, so I ended up testing the RAG capability with gemini-cli separately (e.g., asking it directly to answer questions with RAG).

Sample demand letters (v1 through v4 are not actually improved versions but just simply just four testing runs I conducted with my setup) are included in the sample_demand_letters directory.

Here is what each included file is for.

database_setup_sample_post.sql - This is a modified version of the provided SQL database 
10 minutes ago
gemini_rag_server.py
Add files via upload
10 minutes ago
prepare_for_faiss.py
Add files via upload
10 minutes ago
rag_server.py
Add files via upload
10 minutes ago
rag_with_doc_classification.py
Add files via upload
10 minutes ago
rag_with_gemini.py
Add files via upload
10 minutes ago
requirements.txt
Add files via upload
4 minutes ago
server.py
Add files via upload
10 minutes ago
settings.json
Update settings.json



Note on embedding providers.
- Use EMBEDDING_PROVIDER="huggingface" if you want to use specialized embeddings
- Use EMBEDDING_PROVIDER="together" if you want to use standard open-source (but optimized for inference) embeddings with Together.ai
- Use EMBEDDING_PROVIDER="together" if you want to use standard Gemini embeddings

From my review, the differences between the two embedding models are subtle but meaningful in specific contexts. legal-bge-m3 ("huggingface") excels in retrieving nuanced legal and medical details, while the other two provides slightly more specific factual details in some cases ("together").

