# Updated imports to fix deprecation warnings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os

# Use the correct, non-deprecated import
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("Using langchain_huggingface.HuggingFaceEmbeddings")
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("Using langchain_community.embeddings.HuggingFaceEmbeddings")
    except ImportError:
        print("Error: Could not import HuggingFaceEmbeddings. Please install langchain-huggingface or langchain-community")
        raise

# Initialize the BGE embeddings model with error handling
try:
    bge_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # important for cosine similarity
    )
    print("✓ BGE embeddings model initialized successfully")
except Exception as e:
    print(f"❌ Error initializing embeddings model: {e}")
    print("This might be due to missing dependencies. Try:")
    print("pip install sentence-transformers")
    raise

embeddings = bge_model

def load_texts(text_dir):
    """Load text files from directory and return as Document objects"""
    docs = []
    
    if not os.path.exists(text_dir):
        print(f"Directory {text_dir} does not exist")
        return docs
    
    for filename in os.listdir(text_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(text_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        docs.append(Document(
                            page_content=content, 
                            metadata={"source": filename}
                        ))
                        print(f"✓ Loaded document: {filename} ({len(content)} chars)")
            except Exception as e:
                print(f"❌ Error reading file {filename}: {e}")
    
    print(f"✓ Loaded {len(docs)} documents from {text_dir}")
    return docs

def embed_documents(docs, persist_dir="data/chroma_store"):
    """Create embeddings for documents and store in ChromaDB"""
    if not docs:
        print("❌ No documents to embed")
        return None
    
    try:
        # Create text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split documents into chunks
        print("🔄 Splitting documents into chunks...")
        chunks = splitter.split_documents(docs)
        
        # Filter out empty chunks
        chunks = [c for c in chunks if c.page_content.strip()]
        
        if not chunks:
            print("❌ No valid chunks created from documents")
            return None
        
        print(f"✓ Created {len(chunks)} chunks from {len(docs)} documents")
        
        # Show sample chunk for debugging
        if chunks:
            print(f"📄 Sample chunk: {chunks[0].page_content[:200]}...")
        
        # Ensure the persist directory exists
        os.makedirs(persist_dir, exist_ok=True)
        
        # Clear existing collection if it exists to avoid conflicts
        try:
            if os.path.exists(persist_dir):
                import shutil
                shutil.rmtree(persist_dir)
                os.makedirs(persist_dir, exist_ok=True)
                print("🗑️ Cleared existing vector store")
        except Exception as e:
            print(f"⚠️ Could not clear existing vector store: {e}")
        
        print("🔄 Creating vector store...")
        # Create ChromaDB vector store
        db = Chroma.from_documents(
            chunks, 
            embedding=embeddings, 
            persist_directory=persist_dir,
            collection_name="document_embeddings"
        )
        
        # Verify the vector store was created successfully
        try:
            test_results = db.similarity_search("test", k=1)
            print(f"✅ Successfully created vector store with {len(chunks)} chunks")
            print(f"✓ Vector store verification: {len(test_results)} results found")
        except Exception as e:
            print(f"⚠️ Vector store created but verification failed: {e}")
        
        return db
        
    except Exception as e:
        print(f"❌ Error creating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_existing_vectorstore(persist_dir="data/chroma_store"):
    """Load existing ChromaDB vector store"""
    try:
        if not os.path.exists(persist_dir):
            print(f"📁 Vector store directory {persist_dir} does not exist")
            return None
        
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="document_embeddings"
        )
        
        # Test if the vector store has documents
        try:
            test_results = db.similarity_search("test", k=1)
            print(f"✓ Loaded existing vector store with documents")
            return db
        except Exception as e:
            print(f"⚠️ Vector store exists but appears empty: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return None

def get_or_create_vectorstore(text_dir="data/text_outputs", persist_dir="data/chroma_store"):
    """Get existing vector store or create new one from text files"""
    
    # Always create new vector store for reliability
    print("🔄 Creating fresh vector store...")
    docs = load_texts(text_dir)
    if docs:
        db = embed_documents(docs, persist_dir)
        return db
    else:
        print("❌ No documents found to create vector store")
        return None

# For backward compatibility
def create_embeddings():
    """Create and return embeddings instance"""
    return embeddings

if __name__ == "__main__":
    # Test the embedding functionality
    print("🧪 Testing embedding functionality...")
    
    # Test loading documents
    docs = load_texts("data/text_outputs")
    print(f"📚 Found {len(docs)} documents")
    
    # Test creating vector store
    if docs:
        db = embed_documents(docs)
        if db:
            print("✅ Vector store created successfully")
            
            # Test similarity search
            test_query = "skills"
            results = db.similarity_search(test_query, k=3)
            print(f"✅ Similarity search working: found {len(results)} results")
            
            # Show results
            for i, result in enumerate(results):
                print(f"Result {i+1}: {result.page_content[:150]}...")
        else:
            print("❌ Failed to create vector store")
    else:
        print("❌ No documents found for testing")