import faiss
import numpy as np
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import torch
import os
import json
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from collections import defaultdict

# --- Helper for logging to stderr ---
def log_message(message):
    """Prints a message to stderr to avoid interfering with stdout for MCP."""
    print(message, file=sys.stderr, flush=True)

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATA_DIR = os.getenv("DATA_DIR", os.getcwd())
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", 'document_index.faiss')
DOCUMENT_METADATA_PATH = os.getenv("DOCUMENT_METADATA_PATH", 'document_metadata.json')
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")
GEMINI_EMBEDDING_MODEL = 'gemini-embedding-001'
HUGGINGFACE_EMBEDDING_MODEL = 'yuriyvnv/legal-bge-m3'
GEMINI_LLM_MODEL = 'gemini-2.0-flash-exp'
CHUNK_SIZE = 750
CHUNK_OVERLAP = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Initialization ---
if not GOOGLE_API_KEY:
    log_message("Error: GOOGLE_API_KEY environment variable not set.")
    sys.exit(1)

client = genai.Client(api_key=GOOGLE_API_KEY)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

log_message(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    log_message("WARNING: No GPU detected. Operations will be slower.")
log_message(f"Using embedding provider: {EMBEDDING_PROVIDER}")
log_message(f"Data directory: {DATA_DIR}")

# --- 1. Initialize Embedding Model ---
if EMBEDDING_PROVIDER == "huggingface":
    log_message(f"Loading Hugging Face embedding model: {HUGGINGFACE_EMBEDDING_MODEL}...")
    embedding_model = SentenceTransformer(HUGGINGFACE_EMBEDDING_MODEL, device=DEVICE)
    test_embedding = embedding_model.encode(["Test"], convert_to_numpy=True)
    embedding_dimension = test_embedding.shape[1]
else:
    log_message(f"Using Gemini embedding model: {GEMINI_EMBEDDING_MODEL}...")
    test_response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=["Test"]
    )
    embedding_dimension = len(test_response.embeddings[0].values)
log_message(f"Embedding dimension: {embedding_dimension}")

# --- 2. Load and Chunk Documents ---
def load_and_chunk_documents(data_dir, chunk_size, chunk_overlap):
    document_store = {}
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True
    )

    log_message(f"Loading documents from {data_dir}...")
    if not os.path.exists(data_dir):
        log_message(f"Error: Data directory '{data_dir}' not found.")
        return {}, [], []

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                page_starts = {int(m.group(1)): m.end() for m in re.finditer(r'\[PAGE (\d+)\]', content)}
                sorted_pages = sorted(page_starts.keys())
                page_boundaries = {p: (page_starts[p], page_starts[sorted_pages[i + 1]] if i + 1 < len(sorted_pages) else len(content)) for i, p in enumerate(sorted_pages)}

                docs = text_splitter.create_documents([content])
                log_message(f"  Processed '{filename}': split into {len(docs)} chunks.")

                for i, doc in enumerate(docs):
                    doc_id = f"file_{os.path.basename(filename).replace('.', '_')}_chunk_{i}"
                    start_idx = doc.metadata.get('start_index', 0)
                    end_idx = start_idx + len(doc.page_content)
                    overlapping_pages = [p for p, (p_start, p_end) in page_boundaries.items() if max(start_idx, p_start) < min(end_idx, p_end)]

                    document_store[doc_id] = {
                        "text": doc.page_content,
                        "original_source": filepath,
                        "chunk_index": i,
                        "metadata": {"source_file": filename, "length": len(doc.page_content)},
                        "pages": overlapping_pages
                    }
            except Exception as e:
                log_message(f"  Error reading or processing {filepath}: {e}")

    sorted_docs = sorted(document_store.items(), key=lambda item: (item[1]['original_source'], item[1]['chunk_index']))
    documents_for_faiss_indexing = [info["text"] for _, info in sorted_docs]
    document_ids_for_faiss_indexing = [doc_id for doc_id, _ in sorted_docs]

    log_message(f"Total chunks loaded: {len(documents_for_faiss_indexing)}")
    return document_store, documents_for_faiss_indexing, document_ids_for_faiss_indexing

# --- 3-5. Build Index if it Doesn't Exist ---
if not os.path.exists(FAISS_INDEX_PATH):
    log_message("No FAISS index found. Building a new one...")
    document_store, docs_to_index, doc_ids_to_index = load_and_chunk_documents(DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)

    if not docs_to_index:
        log_message("No documents found to index. Exiting.")
        sys.exit(1)

    log_message("Generating embeddings...")
    document_embeddings = []
    batch_size = 100
    
    if EMBEDDING_PROVIDER == "huggingface":
        for i in range(0, len(docs_to_index), batch_size):
            batch_embeddings = embedding_model.encode(docs_to_index[i:i + batch_size], convert_to_numpy=True)
            document_embeddings.extend(batch_embeddings)
    else:
        for i in range(0, len(docs_to_index), batch_size):
            batch = docs_to_index[i:i + batch_size]
            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=batch
            )
            batch_embeddings = [embedding.values for embedding in response.embeddings]
            document_embeddings.extend(batch_embeddings)
    
    document_embeddings = np.array(document_embeddings).astype('float32')
    log_message(f"Generated {document_embeddings.shape[0]} embeddings.")

    log_message("Creating and saving FAISS index...")
    index_cpu = faiss.IndexFlatL2(embedding_dimension)
    index_cpu.add(document_embeddings)
    faiss.write_index(index_cpu, FAISS_INDEX_PATH)

    log_message("Saving document metadata...")
    faiss_id_to_doc_id = {i: doc_id for i, doc_id in enumerate(doc_ids_to_index)}
    with open(DOCUMENT_METADATA_PATH, 'w') as f:
        json.dump({"document_store": document_store, "faiss_id_to_doc_id": faiss_id_to_doc_id}, f, indent=4)
    log_message("Index and metadata saved.")

# --- 6. Load Index and Metadata ---
log_message("\n--- Loading index and metadata from disk ---")
loaded_index_cpu = faiss.read_index(FAISS_INDEX_PATH)
if DEVICE == "cuda":
    res = faiss.StandardGpuResources()
    loaded_index = faiss.index_cpu_to_gpu(res, 0, loaded_index_cpu)
    log_message("FAISS index moved to GPU.")
else:
    loaded_index = loaded_index_cpu
    log_message("FAISS index remains on CPU.")
log_message(f"Number of vectors in loaded index: {loaded_index.ntotal}")

with open(DOCUMENT_METADATA_PATH, 'r') as f:
    loaded_data = json.load(f)
loaded_document_store = loaded_data["document_store"]
loaded_faiss_id_to_doc_id = loaded_data["faiss_id_to_doc_id"]
log_message("Document metadata loaded.")

# --- 7. RAG Function ---
def answer_question_with_rag(query, faiss_index, doc_store, faiss_id_map, k_retrieval=5):
    # Generate query embedding
    if EMBEDDING_PROVIDER == "huggingface":
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0].reshape(1, -1).astype('float32')
    else:
        response = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=[query]
        )
        query_embedding = np.array([response.embeddings[0].values]).astype('float32')

    _, faiss_indices = faiss_index.search(query_embedding, k_retrieval)

    retrieved_contexts = []
    source_pages = defaultdict(set)
    for faiss_idx in faiss_indices[0]:
        doc_id = faiss_id_map.get(str(faiss_idx))
        if doc_id in doc_store:
            doc_info = doc_store[doc_id]
            retrieved_contexts.append(doc_info["text"])
            source = doc_info["metadata"]["source_file"]
            source_pages[source].update(doc_info.get("pages", []))

    if not retrieved_contexts:
        context_string = ""
        citations = ["No documents referenced."]
    else:
        context_string = "\n".join(retrieved_contexts)
        def pages_to_str(pages):
            if not pages: return ""
            pages = sorted(list(pages))
            ranges = []
            start = prev = pages[0]
            for p in pages[1:]:
                if p != prev + 1:
                    ranges.append(str(start) if start == prev else f"{start}-{prev}")
                    start = p
                prev = p
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            return ", ".join(ranges)
        citations = [f"- {source} (pages {pages_to_str(pages_set)})" if pages_set else f"- {source}" for source, pages_set in source_pages.items()]

    # Generate answer using Gemini
    try:
        system_prompt = "You are a helpful assistant. Use the following context to answer the question. If the answer is not in the context, say 'I don't know'. Do not tell the user you are referencing context."
        user_prompt = f"Context: {context_string}\n\nQuestion: {query}"
        
        response = client.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=[system_prompt, user_prompt],
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=2048
            )
        )

        final_answer = response.text.strip()
        citation_text = "\n\n**Citations:**\n" + "\n".join(citations)
        return final_answer + citation_text
    except Exception as e:
        log_message(f"Error during LLM generation: {e}")
        return "I apologize, but an error occurred while generating the answer."

# --- 8. MCP Protocol Implementation ---
def send_response(response):
    """Send a JSON-RPC response to stdout"""
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()

def send_error(request_id, code, message):
    """Send a JSON-RPC error response"""
    error_response = {
        "jsonrpc": "2.0",
        "id": request_id if request_id is not None else 0,
        "error": {
            "code": code,
            "message": message
        }
    }
    send_response(error_response)

log_message("🚀 Gemini RAG server is starting. Ready for MCP requests from gemini-cli.")

# --- 9. Main Server Loop ---
while True:
    try:
        line = sys.stdin.readline()
        if not line:
            break
        
        log_message(f"Raw request: {line.strip()}")
        
        request = json.loads(line.strip())
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        
        log_message(f"Parsed request - Method: {method}, ID: {request_id}")
        
        is_notification = request_id is None
        
        if method == "initialize":
            if is_notification:
                log_message("Error: initialize must have an id")
                continue
                
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "gemini-rag-legal-docs",
                        "version": "1.0.0"
                    }
                }
            }
            send_response(response)
            
        elif method == "initialized":
            log_message("MCP server initialized successfully")
            continue
            
        elif method == "tools/list":
            if is_notification:
                log_message("Error: tools/list must have an id")
                continue
                
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "query_documents",
                            "description": "Search and query legal documents using Gemini-powered RAG (Retrieval-Augmented Generation)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The question or search query to ask about the documents"
                                    },
                                    "k_retrieval": {
                                        "type": "integer",
                                        "description": "Number of document chunks to retrieve (default: 5)",
                                        "default": 5,
                                        "minimum": 1,
                                        "maximum": 20
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    ]
                }
            }
            send_response(response)
            
        elif method == "tools/call":
            if is_notification:
                log_message("Error: tools/call must have an id")
                continue
                
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "query_documents":
                query = arguments.get("query")
                k_retrieval = arguments.get("k_retrieval", 5)
                
                if not query:
                    send_error(request_id, -32602, "Missing required parameter: query")
                    continue
                
                try:
                    log_message(f"Processing query: {query}")
                    answer = answer_question_with_rag(
                        query, 
                        loaded_index, 
                        loaded_document_store, 
                        loaded_faiss_id_to_doc_id, 
                        k_retrieval=k_retrieval
                    )
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": answer
                                }
                            ]
                        }
                    }
                    send_response(response)
                    
                except Exception as e:
                    log_message(f"Error processing query: {e}")
                    send_error(request_id, -32603, f"Internal error: {str(e)}")
            else:
                send_error(request_id, -32601, f"Unknown tool: {tool_name}")
                
        elif method == "ping":
            if is_notification:
                log_message("Error: ping must have an id")
                continue
                
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {}
            }
            send_response(response)
            
        else:
            if not is_notification:
                send_error(request_id, -32601, f"Method not found: {method}")
            else:
                log_message(f"Ignoring unknown notification: {method}")

    except json.JSONDecodeError as e:
        log_message(f"Error: Received invalid JSON: {e}")
        send_error(None, -32700, "Parse error")
    except Exception as e:
        log_message(f"An unexpected error occurred: {e}")
        try:
            req_id = request_id
        except NameError:
            req_id = None
        send_error(req_id, -32603, f"Internal error: {str(e)}")
