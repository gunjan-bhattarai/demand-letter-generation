import faiss
import numpy as np
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import torch
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import re
from collections import defaultdict

# --- Configuration ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")  # Options: 'gemini' or 'huggingface'
GEMINI_EMBEDDING_MODEL = 'gemini-embedding-001'  # Gemini embedding model
HUGGINGFACE_EMBEDDING_MODEL = 'yuriyvnv/legal-bge-m3'  # Hugging Face legal-specific model
GEMINI_LLM_MODEL = 'gemini-2.5-flash'  # Gemini LLM model
FAISS_INDEX_PATH = 'document_index.faiss'
DOCUMENT_METADATA_PATH = 'document_metadata.json'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Gemini client (requires GOOGLE_API_KEY environment variable)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Ensure this is set in your environment
client = genai.Client(api_key=GOOGLE_API_KEY)

# --- New Configuration for Text File Loading and Chunking ---
DATA_DIR = os.getcwd()
CHUNK_SIZE = 750
CHUNK_OVERLAP = 50

print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: No GPU detected. FAISS will run on CPU, which will be much slower.")
print(f"Using embedding provider: {EMBEDDING_PROVIDER}")

# --- 1. Initialize Embedding Model ---
if EMBEDDING_PROVIDER == "huggingface":
    print(f"Loading Hugging Face embedding model: {HUGGINGFACE_EMBEDDING_MODEL}...")
    embedding_model = SentenceTransformer(HUGGINGFACE_EMBEDDING_MODEL, device=DEVICE)
    # Test embedding to get dimension
    test_embedding = embedding_model.encode(["Test sentence to get embedding dimension."], convert_to_numpy=True)
    embedding_dimension = test_embedding.shape[1]
    print(f"Embedding dimension: {embedding_dimension}")
else:
    print(f"Using Gemini embedding model: {GEMINI_EMBEDDING_MODEL}...")
    test_response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=["Test sentence to get embedding dimension."]
    )
    embedding_dimension = len(test_response.embeddings[0].values)
    print(f"Embedding dimension: {embedding_dimension}")

# --- 2. Load and Chunk Documents from Text Files ---
def load_and_chunk_documents(data_dir, chunk_size, chunk_overlap):
    documents_data = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    print(f"Loading documents from {data_dir} and chunking with size {chunk_size}, overlap {chunk_overlap}...")
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found. Please create it and place your text files there.")
        return [], {}, []

    all_file_texts = []
    current_doc_id = 0
    document_store = {}

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse page markers to get page boundaries
                page_starts = {}
                for match in re.finditer(r'\[PAGE (\d+)\]', content):
                    page_num = int(match.group(1))
                    page_starts[page_num] = match.end()

                sorted_pages = sorted(page_starts.keys())
                page_boundaries = {}
                for i, p in enumerate(sorted_pages):
                    start = page_starts[p]
                    end = page_starts[sorted_pages[i + 1]] if i + 1 < len(sorted_pages) else len(content)
                    page_boundaries[p] = (start, end)

                # Split into chunks with metadata
                docs = text_splitter.create_documents([content])
                chunks = [d.page_content for d in docs]
                start_indices = [d.metadata.get('start_index', 0) for d in docs]

                print(f"  Processed '{filename}': split into {len(chunks)} chunks.")

                for i, chunk_text in enumerate(chunks):
                    doc_id = f"file_{os.path.basename(filename).replace('.', '_')}_chunk_{i}"
                    start_idx = start_indices[i]
                    end_idx = start_idx + len(chunk_text)

                    overlapping_pages = [p for p, (p_start, p_end) in page_boundaries.items() if max(start_idx, p_start) < min(end_idx, p_end)]

                    document_store[doc_id] = {
                        "text": chunk_text,
                        "original_source": filepath,
                        "chunk_index": i,
                        "metadata": {"source_file": filename, "length": len(chunk_text)},
                        "pages": overlapping_pages
                    }
                    all_file_texts.append(chunk_text)

            except Exception as e:
                print(f"  Error reading or processing {filepath}: {e}")

    documents_for_faiss_indexing = [doc_info["text"] for doc_id, doc_info in sorted(document_store.items(), key=lambda item: (item[1]['original_source'], item[1]['chunk_index']))]
    document_ids_for_faiss_indexing = [doc_id for doc_id, doc_info in sorted(document_store.items(), key=lambda item: (item[1]['original_source'], item[1]['chunk_index']))]

    print(f"Total chunks loaded: {len(documents_for_faiss_indexing)}")
    return document_store, documents_for_faiss_indexing, document_ids_for_faiss_indexing

# Check if FAISS index exists
index_present = any(filename.endswith(".faiss") for filename in os.listdir(DATA_DIR))

if not index_present:
    print("No FAISS index found. Loading and chunking documents...")
    loaded_document_store, documents_for_faiss_indexing, document_ids_for_faiss_indexing = load_and_chunk_documents(DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)

    if not documents_for_faiss_indexing:
        print("No documents loaded or processed. Exiting.")
        exit()

    # --- 3. Generate Embeddings ---
    print("Generating embeddings for documents...")
    document_embeddings = []
    if EMBEDDING_PROVIDER == "huggingface":
        print(f"Using {HUGGINGFACE_EMBEDDING_MODEL} for embeddings...")
        # Process in batches to manage memory
        batch_size = 100
        for i in range(0, len(documents_for_faiss_indexing), batch_size):
            batch = documents_for_faiss_indexing[i:i + batch_size]
            batch_embeddings = embedding_model.encode(batch, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)
            document_embeddings.extend(batch_embeddings)
    else:
        print(f"Using Gemini for embeddings...")
        batch_size = 100
        for i in range(0, len(documents_for_faiss_indexing), batch_size):
            batch = documents_for_faiss_indexing[i:i + batch_size]
            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=batch
            )
            batch_embeddings = [embedding.values for embedding in response.embeddings]
            document_embeddings.extend(batch_embeddings)
    
    document_embeddings = np.array(document_embeddings).astype('float32')
    print(f"Generated {document_embeddings.shape[0]} embeddings of shape {document_embeddings.shape[1]}.")

    # --- 4. FAISS Index Creation and Population ---
    print("Creating FAISS index...")
    index_cpu = faiss.IndexFlatL2(embedding_dimension)
    index_cpu.add(document_embeddings)

    if DEVICE == "cuda":
        print("Moving FAISS index to GPU...")
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        index = gpu_index
        print("FAISS index moved to GPU successfully.")
    else:
        index = index_cpu
        print("FAISS index remains on CPU.")

    print(f"Number of vectors in FAISS index: {index.ntotal}")

    # --- 5. Saving the FAISS Index and Document Metadata ---
    print(f"\nSaving FAISS index to {FAISS_INDEX_PATH}...")
    if DEVICE == "cuda":
        cpu_index_to_save = faiss.index_gpu_to_cpu(index)
        faiss.write_index(cpu_index_to_save, FAISS_INDEX_PATH)
    else:
        faiss.write_index(index, FAISS_INDEX_PATH)
    print("FAISS index saved.")

    print(f"Saving document metadata to {DOCUMENT_METADATA_PATH}...")
    faiss_id_to_doc_id = {i: doc_id for i, doc_id in enumerate(document_ids_for_faiss_indexing)}
    with open(DOCUMENT_METADATA_PATH, 'w') as f:
        json.dump({"document_store": loaded_document_store, "faiss_id_to_doc_id": faiss_id_to_doc_id}, f, indent=4)
    print("Document metadata saved.")

# --- 6. Loading the FAISS Index and Document Metadata ---
print("\n--- Loading from disk ---")
print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
loaded_index_cpu = faiss.read_index(FAISS_INDEX_PATH)

if DEVICE == "cuda":
    print("Moving loaded FAISS index to GPU...")
    res_loaded = faiss.StandardGpuResources()
    loaded_index = faiss.index_cpu_to_gpu(res_loaded, 0, loaded_index_cpu)
    print("FAISS index moved to GPU successfully.")
else:
    loaded_index = loaded_index_cpu
    print("FAISS index remains on CPU.")

print(f"Number of vectors in loaded FAISS index: {loaded_index.ntotal}")

print(f"Loading document metadata from {DOCUMENT_METADATA_PATH}...")
with open(DOCUMENT_METADATA_PATH, 'r') as f:
    loaded_data = json.load(f)
loaded_document_store = loaded_data["document_store"]
loaded_faiss_id_to_doc_id = loaded_data["faiss_id_to_doc_id"]
print("Document metadata loaded.")

# --- 7. Define RAG Function ---
def answer_question_with_rag(query, faiss_index, doc_store, faiss_id_map, k_retrieval=5):
    print(f"\nUser Query: {query}")

    # 1. Retrieve relevant documents using FAISS
    if EMBEDDING_PROVIDER == "huggingface":
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0].astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
    else:
        response = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=[query]
        )
        query_embedding = np.array([response.embeddings[0].values]).astype('float32')

    distances, faiss_indices = faiss_index.search(query_embedding, k_retrieval)

    retrieved_contexts = []
    source_pages = defaultdict(set)
    for rank, faiss_idx in enumerate(faiss_indices[0]):
        doc_id = faiss_id_map.get(str(faiss_idx))
        if doc_id and doc_id in doc_store:
            document_info = doc_store[doc_id]
            retrieved_contexts.append(document_info["text"])
            source = document_info["metadata"]["source_file"]
            pages = document_info.get("pages", [])
            source_pages[source].update(pages)
        else:
            print(f"  Warning: FAISS index {faiss_idx} not found in document store mapping.")

    if not retrieved_contexts:
        print("No relevant documents found. Answering based on LLM's general knowledge.")
        context_string = ""
        citations = ["No documents referenced."]
    else:
        context_string = "\n".join(retrieved_contexts)

        def pages_to_str(pages):
            if not pages:
                return ""
            pages = sorted(pages)
            ranges = []
            start = pages[0]
            prev = pages[0]
            for p in pages[1:]:
                if p == prev + 1:
                    prev = p
                else:
                    ranges.append(str(start) if start == prev else f"{start}-{prev}")
                    start = p
                    prev = p
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            return ", ".join(ranges)

        citations = []
        for source, pages_set in source_pages.items():
            pages_str = pages_to_str(list(pages_set))
            citation = f"- {source}"
            if pages_str:
                citation += f" (pages {pages_str})"
            citations.append(citation)

    # 2. Construct the prompt for Gemini
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the following context to answer the question. If the answer is not in the context, say 'I don't know'. Do not tell the user you are referencing context - make it sound as if you are speaking based on your own knowledge."
        },
        {
            "role": "user",
            "content": f"Context: {context_string}\n\nQuestion: {query}"
        }
    ]

    # 3. Generate the answer using Gemini
    try:
        response = client.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=[messages[0]["content"], messages[1]["content"]],
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=2048
            )
        )

        final_answer = response.text.strip().split("</think>")[-1]
        citation_text = "\n\n**Citations:**\n" + "\n".join(citations)
        final_answer_with_citations = final_answer + citation_text

        print("\n--- LLM Answer ---")
        print(final_answer_with_citations)
        return final_answer_with_citations

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        print("Could not generate an answer.")
        return "I apologize, but I could not generate an answer at this time."

client_name = "John Smith"

questions = [
    "What is the admission of fault by the insured, including any details about distraction or cause?",
    "Extract the liability limits, property damage limits, and policy status.",
    f"What are the vehicle damage details for {client_name}'s car, including specific components affected?",
    "Who is the assigned claims adjuster and their contact information?",
    "What are the claim number, policy number, and insured's name?",
    f"What is {client_name}'s hourly rate, overtime rate, regular schedule, and average weekly earnings pre-accident?",
    "Calculate or extract the total gross earnings and average weekly gross for the pre-accident period (January 1 to March 15, 2024).",
    "What employment verification details are provided, including position, employer, and hire date?",
    "List the detailed payroll records by pay period, including regular hours, overtime, gross pay, and net pay.",
    "How does describe the wage loss due to the accident, including any estimates or averages?",
    "Describe the sequence of events and narrative of the accident per any available police reports, including impact details and speed estimates.",
    "What are the driver statements from all involved parties per any available police reports?",
    "Extract all vehicle information (make, model, VIN, plate, damage) for both vehicles involved.",
    "What are the accident location, date, time, weather, road conditions, and traffic control details?",
    "Who are the parties involved (names, DOBs, licenses, addresses, phones, insurance), and any fault indicators?",
    f"List all injuries sustained by {client_name}, including descriptions from initial exam and follow-ups.",
    "What are the total medical expenses paid to date, broken down by category (e.g., ER, MRI, therapy)?",
    "Describe the treatment plan, medications, physical therapy, and work restrictions across all visits.",
    "What is the prognosis, permanent impairment rating, future medical needs, and estimated future costs?",
    "Extract pain severity ratings, symptom descriptions (e.g., back pain, stiffness), and impact on daily life or work."
]

for q in questions:
    answer_question_with_rag(q, loaded_index, loaded_document_store, loaded_faiss_id_to_doc_id, k_retrieval=5)
    print("\n" + "="*50 + "\n")
