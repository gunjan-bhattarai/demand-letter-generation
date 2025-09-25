# demand-letter-generation
Demand letter generation using MCP servers and gemini-cli (with RAG implementation as well).

Two MCP servers were ultimately created - one for querying the database for case details, plaintiff info, defendant info, etc. and the other to conduct traditional retrieval augmented generation (RAG). I ended up converting my RAG code into an MCP server in order to allow gemini-cli (my MCP client of choice for this project) to access it. However, Gemini repeatedly preferred to use its own ReadFile tool in order to acquire the needed information, so I ended up testing the RAG capability with gemini-cli separately (e.g., asking it directly to answer questions with RAG).

Sample demand letters (v1 through v4 are not actually improved versions but just simply just four testing runs I conducted with my setup) are included in the sample_demand_letters directory.

Here is what each included file is for.

database_setup_sample_post.sql - This is a modified version of the provided SQL database to remove syntax errors when I tried to query it with PostgreSQL.

gemini_rag_server.py - MCP server for RAG that used Gemini for LLM

prepare_for_faiss.py - Preprocessing code that converted PDFs to .txt so I could chunk them with FAISS (Facebook AI Similarity Search, which is what many vector databases are built on top of)

rag_server.py - MCP server for RAG that used Together.ai for LLM (I defaulted to GLM-4.5-Air, but this wasn't based on optimality - I just wanted to test an open source model to validate my approach didn't just work for proprietary models).

rag_with_doc_classification.py - Modified RAG script that adds document classification as well as some specialized domain-based chunking (differing sizes of chunks based on domain classification and adding headers of documents to chunks). Performed around or slightly worse than my standard chunking approach so I opted to just use that instead for the MCP server. Includes 20 sample questions at the end for testing. Includes 20 sample questions at the end for testing.

rag_with_gemini.py - Modified RAG script that uses Gemini instead of Together.ai models. Continues to offer the use of an open-source Huggingface model to offer embeddings specifically catering to the legal domain (see note on embedding providers at the end). Includes 20 sample questions at the end for testing.

requirements.txt - I ran pip freeze > requirements.txt on what libraries ended up being downloaded in my virtual environment during this project.

server.py - MCP server for querying database_setup_sample_post.sql.

settings.json - gemini-cli's settings.json when using the Gemini RAG server.

To run, move the settings.json of choice to ~/.gemini/settings.json and run the command "gemini" (assuming you have downloaded the gemini-cli already). You can test the approach with the following prompt:

"Generate a demand letter for Case 2024-PI-001. Include all medical expenses, lost wages, and pain and suffering damages. Reference specific medical findings from Dr. Jones and cite the police report for liability determination. The demand should be professional and include proper legal citations."

Note on embedding providers.
- Use EMBEDDING_PROVIDER="huggingface" if you want to use specialized embeddings
- Use EMBEDDING_PROVIDER="together" if you want to use standard open-source (but optimized for inference) embeddings with Together.ai
- Use EMBEDDING_PROVIDER="together" if you want to use standard Gemini embeddings

From my review, the differences between the two embedding models are subtle but meaningful in specific contexts. legal-bge-m3 ("huggingface") excels in retrieving nuanced legal and medical details, while the other two provides slightly more specific factual details in some cases ("together").

