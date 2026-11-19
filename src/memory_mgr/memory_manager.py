import os
import json
import hashlib
import chromadb
from chromadb.utils import embedding_functions


# *******************************************************
# Initialize ChromaDB client and collection 
# *******************************************************

DB_PATH = "./ipac_storage_db"
MANIFEST_PATH = os.path.join(DB_PATH, "index_manifest.json")

REPORT_TO_WRITE = "Summarized_Business_Findings.md"
HTML_TO_WRITE = "Summarized_Business_Findings.html"

# 1. Initialize persistent storage and Ollama embeddings
chroma_client = chromadb.PersistentClient(path=DB_PATH)

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text",
)

collection = chroma_client.get_or_create_collection(
    name="ipac_codebase_knowledge",
    embedding_function=ollama_ef,
)

# ALLOWED_EXTENSIONS = {'**/*'}
#    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", 
#    ".java", ".yaml", ".yml", ".json", ".toml", ".md", ".sh", ".env"
#}

# IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "chroma_db", "dist", "build"}

IGNORE_DIRS = {'.git', 'node_modules', 'webpack', 'build', 'dist','.gitignore','.dockerignore','i18n',
                'build', 'buildspec', '.venv','.env', 'venv','ipac_storage_db','__pycache__','.github','.agents','taglib'
}
#                '**/.devcontainer/**', '**/bin/**', '**/lib/**', '**/assets/**',
#                '**/build/**', '**/buildspec/**', '**/scripts/**', '**/.gradle/**', 
#                '**/.git/**', '**/.gitignore/**', '**.dockerignore/**', '**/.gradle/**', 
#                '**/.idea/**', '**/.vscode/**', '**/node_modules/**', '**/.gsp/**', '**/assets/**', 
#                '**/domain/**', '**/i18n/**', '**/taglib/**', '**/.env/**', '**/.venv/**', "**/venv/**", 
#                "**/env/**", "**/__pycache__/**"'.github/**','**/.agents/**','**/ipac_storage/**'}


def load_manifest() -> dict:
    """Loads the file-hash manifest from disk."""
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(manifest: dict) -> None:
    """Saves the updated file-hash manifest to disk."""
    os.makedirs(DB_PATH, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def get_file_hash(content: str) -> str:
    """Computes an MD5 hash of the file content."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """Splits text into overlapping character chunks to preserve code context."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def index_codebase(target_dirs: str | list[str]) -> None:
    """Walks target directories, skips unchanged files via hash caching, and updates ChromaDB."""
    if isinstance(target_dirs, str):
        target_dirs = [target_dirs]

    print("📚 Checking codebase changes for ChromaDB...")
    manifest = load_manifest()
    seen_files = set()

    indexed_files = 0
    skipped_files = 0
    total_chunks_upserted = 0

    for base_dir in target_dirs:
        if not os.path.exists(base_dir):
            print(f"⚠️ Warning: Directory '{base_dir}' does not exist. Skipping.")
            continue

        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                # if ext not in ALLOWED_EXTENSIONS:
                #     continue

                filepath = os.path.normpath(os.path.join(root, file))
                seen_files.add(filepath)

                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()

                    if not content:
                        continue

                    current_hash = get_file_hash(content)

                    # Skip if file content hasn't changed since last index
                    if filepath in manifest and manifest[filepath] == current_hash:
                        skipped_files += 1
                        continue

                    # Remove old chunks for this file first (in case the file shrank)
                    collection.delete(where={"source": filepath})

                    chunks = chunk_text(content)
                    ids = []
                    documents = []
                    metadatas = []

                    for idx, chunk in enumerate(chunks):
                        raw_id = f"{filepath}::chunk_{idx}"
                        doc_id = hashlib.md5(raw_id.encode()).hexdigest()
                        formatted_chunk = f"File: {filepath} (Part {idx + 1}/{len(chunks)})\n\n{chunk}"

                        ids.append(doc_id)
                        documents.append(formatted_chunk)
                        metadatas.append({
                            "source": filepath,
                            "chunk_index": idx,
                            "type": "code_chunk"
                        })

                    if documents:
                        collection.upsert(
                            ids=ids, 
                            documents=documents, 
                            metadatas=metadatas
                            )
                        
                        total_chunks_upserted += len(documents)

                    # Update manifest hash
                    manifest[filepath] = current_hash
                    indexed_files += 1

                except Exception as e:
                    print(f"⚠️ Failed to index {filepath}: {e}")

    # Clean up deleted files from both ChromaDB and the manifest
    deleted_files = [fp for fp in list(manifest.keys()) if fp not in seen_files]
    for fp in deleted_files:
        collection.delete(where={"source": fp})
        del manifest[fp]

    save_manifest(manifest)

    results = [    
        f"✅ Index sync complete! "
        f"Updated: {indexed_files} files ({total_chunks_upserted} chunks) | "
        f"Unchanged: {skipped_files} files | "
        f"Removed: {len(deleted_files)} deleted files.\n"
    ]
        
    print(results)

    content = "\n".join(results)

    with open(REPORT_TO_WRITE, "w", encoding="utf-8") as file: file.write(content)


# --- TOOLS FOR THE AGENT ---

def sub_query_knowledge_base(query_text: str, n_results: int = 2) -> str:

    results = collection.query(
        query_texts=[query_text],
        n_results=int(n_results)
    )
    documents = results.get("documents", [[]])[0]
    if not documents:
        return "No matching records found in ChromaDB."
    return "\n\n====================\n\n".join(documents)


def sub_save_to_knowledge_base(doc_id: str, summary_text: str, category: str = "architecture_note") -> str:

    collection.upsert(
        ids=[str(doc_id)],
        documents=[f"[AGENT NOTE - {category.upper()}]\n{summary_text}"],
        metadatas=[{"source": "agent_memory", "type": category}]
    )
    return f"Saved architectural note '{doc_id}' to ChromaDB."
