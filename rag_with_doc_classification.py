import faiss
import numpy as np
from together import Together
from sentence_transformers import SentenceTransformer
import torch
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from collections import defaultdict

# --- Configuration ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "together")
TOGETHER_EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
HUGGINGFACE_EMBEDDING_MODEL = 'yuriyvnv/legal-bge-m3'
FAISS_INDEX_PATH = 'document_index.faiss'
DOCUMENT_METADATA_PATH = 'document_metadata.json'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOGETHER_API_KEY = "9cb7f8d755d697c8ef0a574446ed1b5219ec948422508f256f2d73699c72e5d6"
client = Together(api_key=TOGETHER_API_KEY)
MODEL_NAME = "zai-org/GLM-4.5-Air-FP8"

# --- New Configuration for Text File Loading and Chunking ---
DATA_DIR = os.getcwd()
DEFAULT_CHUNK_SIZE = 750
DEFAULT_CHUNK_OVERLAP = 50

# Document type-specific chunking parameters
DOC_TYPE_PARAMS = {
    'medical': {'chunk_size': 500, 'chunk_overlap': 100},  # Smaller chunks for detailed medical notes
    'police': {'chunk_size': 1000, 'chunk_overlap': 150},  # Larger chunks for narratives
    'insurance': {'chunk_size': 750, 'chunk_overlap': 100},
    'employment': {'chunk_size': 600, 'chunk_overlap': 100},  # Moderate for structured payroll data
    'general': {'chunk_size': DEFAULT_CHUNK_SIZE, 'chunk_overlap': DEFAULT_CHUNK_OVERLAP}
}

# Patterns for section detection
SECTION_PATTERNS = [
    r'^(CC:|HPI:|PE:|A/P:|Chief Complaint|History of Present Illness|Physical Examination|Assessment|Plan|Prognosis)\s*',  # Medical SOAP notes
    r'^(INITIAL EXAM(INATION)?|FOLLOW[- ]?UP VISIT|FINAL EVAL(UATION)?|MEDICAL RECORDS?)\s*-?\s*(.*)$',
    r'^(TRAFFIC COLLISION REPORT|POLICE REPORT|INCIDENT REPORT|NARRATIVE|VEHICLES? INVOLVED|DRIVER STATEMENTS?|FAULT DETERMINATION).*$',
    r'^(INSURANCE CLAIM|POLICY #|CLAIM #|ADJUSTER).*$',
    r'^(WAGE STATEMENT|PAYROLL|EARNINGS|HOURLY RATE).*$',
    r'^(\d+\.(?:\d+\.)*)\s*(.+?)$',  # Numbered sections
    r'^([A-Z][A-Z\s]{2,}:?)$',  # ALL CAPS headers
]

print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: No GPU detected. FAISS and LLM will run on CPU, which will be much slower.")
print(f"Using embedding provider: {EMBEDDING_PROVIDER}")

# --- 1. Initialize Embedding Model ---
if EMBEDDING_PROVIDER == "huggingface":
    print(f"Loading Hugging Face embedding model: {HUGGINGFACE_EMBEDDING_MODEL}...")
    embedding_model = SentenceTransformer(HUGGINGFACE_EMBEDDING_MODEL, device=DEVICE)
    test_embedding = embedding_model.encode(["Test sentence to get embedding dimension."], convert_to_numpy=True)
    embedding_dimension = test_embedding.shape[1]
    print(f"Embedding dimension: {embedding_dimension}")
else:
    print(f"Using Together AI embedding model: {TOGETHER_EMBEDDING_MODEL}...")
    test_response = client.embeddings.create(
        model=TOGETHER_EMBEDDING_MODEL,
        input=["Test sentence to get embedding dimension."]
    )
    embedding_dimension = len(test_response.data[0].embedding)
    print(f"Embedding dimension: {embedding_dimension}")

# --- 2. Document Type Classification ---
def detect_document_type(content: str) -> str:
    """Classify document based on keyword presence"""
    content_upper = content.upper()
    keyword_counts = {
        'medical': sum(1 for term in ['MEDICAL RECORDS', 'PATIENT:', 'CHIEF COMPLAINT', 'PHYSICAL EXAMINATION', 'DR.', 'MD', 'SOAP', 'HPI:', 'A/P:'] if term in content_upper),
        'police': sum(1 for term in ['POLICE REPORT', 'TRAFFIC COLLISION', 'OFFICER', 'BADGE', 'VEHICLE #', 'NARRATIVE', 'VIOLATION'] if term in content_upper),
        'insurance': sum(1 for term in ['INSURANCE CLAIM', 'POLICY #', 'CLAIM #', 'ADJUSTER'] if term in content_upper),
        'employment': sum(1 for term in ['WAGE STATEMENT', 'PAYROLL', 'EARNINGS', 'HOURLY RATE'] if term in content_upper),
    }
    max_type = max(keyword_counts, key=keyword_counts.get)
    return max_type if keyword_counts[max_type] >= 2 else 'general'  # Require at least 2 matches

# --- 3. Load and Chunk Documents with Type-Specific Parameters ---
def load_and_chunk_documents(data_dir):
    documents_data = []
    document_store = {}
    all_file_texts = []

    print(f"Loading documents from {data_dir} with type-specific chunking...")
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return [], {}, []

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Detect document type
                doc_type = detect_document_type(content)
                chunk_size = DOC_TYPE_PARAMS[doc_type]['chunk_size']
                chunk_overlap = DOC_TYPE_PARAMS[doc_type]['chunk_overlap']
                print(f"  Processing '{filename}' as {doc_type} document (chunk_size={chunk_size}, overlap={chunk_overlap})")

                # Initialize text splitter with type-specific parameters
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    add_start_index=True,
                    separators=['\n\n', '\n', '. ', ' ', '']  # Prioritize natural breaks
                )

                # Parse page markers
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

                # Split into chunks with section awareness
                docs = text_splitter.create_documents([content])
                chunks = []
                current_section = "Document Start"

                for doc in docs:
                    chunk_text = doc.page_content
                    start_idx = doc.metadata.get('start_index', 0)
                    
                    # Check for section headers
                    for pattern in SECTION_PATTERNS:
                        if re.match(pattern, chunk_text.strip(), re.IGNORECASE):
                            current_section = re.match(pattern, chunk_text.strip(), re.IGNORECASE).group(0)
                            break
                    
                    # Ensure section headers are included in chunks
                    if current_section and not chunk_text.startswith(current_section):
                        chunk_text = f"{current_section}\n{chunk_text}"
                    
                    chunks.append({
                        'text': chunk_text,
                        'start_index': start_idx,
                        'section': current_section
                    })

                print(f"  Processed '{filename}': split into {len(chunks)} chunks.")

                for i, chunk in enumerate(chunks):
                    chunk_text = chunk['text']
                    start_idx = chunk['start_index']
                    end_idx = start_idx + len(chunk_text)
                    section = chunk['section']

                    overlapping_pages = [p for p, (p_start, p_end) in page_boundaries.items() if max(start_idx, p_start) < min(end_idx, p_end)]

                    doc_id = f"file_{os.path.basename(filename).replace('.', '_')}_chunk_{i}"
                    document_store[doc_id] = {
                        "text": chunk_text,
                        "original_source": filepath,
                        "chunk_index": i,
                        "metadata": {
                            "source_file": filename,
                            "length": len(chunk_text),
                            "document_type": doc_type,
                            "section": section
                        },
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
    loaded_document_store, documents_for_faiss_indexing, document_ids_for_faiss_indexing = load_and_chunk_documents(DATA_DIR)

    if not documents_for_faiss_indexing:
        print("No documents loaded or processed. Exiting.")
        exit()

    # --- 4. Generate Embeddings ---
    print("Generating embeddings for documents...")
    document_embeddings = []
    if EMBEDDING_PROVIDER == "huggingface":
        print(f"Using {HUGGINGFACE_EMBEDDING_MODEL} for embeddings...")
        batch_size = 100
        for i in range(0, len(documents_for_faiss_indexing), batch_size):
            batch = documents_for_faiss_indexing[i:i + batch_size]
            batch_embeddings = embedding_model.encode(batch, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)
            document_embeddings.extend(batch_embeddings)
    else:
        print(f"Using Together AI for embeddings...")
        batch_size = 100
        for i in range(0, len(documents_for_faiss_indexing), batch_size):
            batch = documents_for_faiss_indexing[i:i + batch_size]
            response = client.embeddings.create(
                model=TOGETHER_EMBEDDING_MODEL,
                input=batch
            )
            batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            document_embeddings.extend(batch_embeddings)
    
    document_embeddings = np.array(document_embeddings).astype('float32')
    print(f"Generated {document_embeddings.shape[0]} embeddings of shape {document_embeddings.shape[1]}.")

    # --- 5. FAISS Index Creation and Population ---
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

    # --- 6. Saving the FAISS Index and Document Metadata ---
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

# --- 7. Loading the FAISS Index and Document Metadata ---
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

# --- 8. Define RAG Function with Type-Aware Context ---
def answer_question_with_rag(query, faiss_index, doc_store, faiss_id_map, k_retrieval=5):
    print(f"\nUser Query: {query}")

    # Retrieve relevant documents using FAISS
    if EMBEDDING_PROVIDER == "huggingface":
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0].astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
    else:
        response = client.embeddings.create(
            model=TOGETHER_EMBEDDING_MODEL,
            input=[query]
        )
        query_embedding = np.array([response.data[0].embedding]).astype('float32')

    distances, faiss_indices = faiss_index.search(query_embedding, k_retrieval)

    retrieved_contexts = []
    source_pages = defaultdict(set)
    for rank, faiss_idx in enumerate(faiss_indices[0]):
        doc_id = faiss_id_map.get(str(faiss_idx))
        if doc_id and doc_id in doc_store:
            document_info = doc_store[doc_id]
            doc_type = document_info["metadata"]["document_type"]
            section = document_info["metadata"]["section"]
            context = f"[{doc_type.title()} - {section}]\n{document_info['text']}"
            retrieved_contexts.append(context)
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
        context_string = "\n\n---\n\n".join(retrieved_contexts)

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

    # Construct the prompt
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specializing in personal injury cases. Use the provided context from medical records, police reports, insurance claims, or employment records to answer the question. If the answer is not in the context, say 'I don't know'. Do not mention you are using context."
        },
        {
            "role": "user",
            "content": f"Context: {context_string}\n\nQuestion: {query}"
        }
    ]

    # Generate the answer
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            stream=False
        )

        final_answer = response.choices[0].message.content.strip().split("</think>")[-1]
        citation_text = "\n\n**Citations:**\n" + "\n".join(citations)
        final_answer_with_citations = final_answer + citation_text

        print("\n--- LLM Answer ---")
        print(final_answer_with_citations)
        return final_answer_with_citations

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        print("Could not generate an answer.")
        return "I apologize, but I could not generate an answer at this time."

# --- 9. Run Queries ---
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
