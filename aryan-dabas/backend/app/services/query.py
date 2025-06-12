import os
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_chroma import Chroma

# Consistent embedding import
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Initialize LLM (Ollama)
try:
    llm = OllamaLLM(model="gemma:2b", temperature=0.2)
except Exception as e:
    print(f"❌ Warning: Failed to initialize Ollama LLM: {e}")
    llm = None

# 💬 Refined Prompt Template for accurate skill extraction
prompt = PromptTemplate.from_template(
    """You are an AI assistant helping extract factual information from resumes.

The user question is about a person's skills. From the context below, extract a full, clean list of **skills, tools, and soft skills** mentioned.

Always quote directly if available. Be accurate and specific. If the context includes a bulleted list, preserve it.

Context:
{context}

Question:
{question}

Answer:"""
)

# Chain connection
chain = prompt | llm if llm else None

def check_ollama_connection():
    try:
        if llm is None:
            return False, "LLM not initialized"
        test = llm.invoke("Hello")
        return True, "Ollama connection successful"
    except Exception as e:
        return False, f"Ollama connection failed: {e}"

def ask_question(query, persist_dir="data/chroma_store"):
    if chain is None:
        return "❌ Ollama not initialized. Please run `ollama run gemma:2b`."

    if not os.path.exists(persist_dir):
        return "❌ No vector DB found. Please embed some documents first."

    try:
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="document_embeddings"
        )

        try:
            collection = db._collection
            count = collection.count()
            if count == 0:
                return "⚠️ No documents found in the database."
        except Exception as e:
            print(f"⚠️ Failed to fetch collection count: {e}")

        # Retrieve more chunks for full skill coverage
        docs = db.similarity_search(query, k=12)

        if not docs:
            return "⚠️ No relevant documents found."

        # Prepare full context
        context = "\n".join([doc.page_content for doc in docs])

        # Ask LLM
        response = chain.invoke({"context": context, "question": query})

        return response.strip() if isinstance(response, str) else str(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error processing query: {e}"

def get_relevant_documents(query, persist_dir="data/chroma_store", k=6):
    try:
        if not os.path.exists(persist_dir):
            print("⚠️ No persist directory.")
            return []

        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="document_embeddings"
        )

        collection = db._collection
        if collection.count() == 0:
            print("⚠️ Collection is empty.")
            return []

        retriever = db.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return []

def debug_database_contents(persist_dir="data/chroma_store"):
    try:
        if not os.path.exists(persist_dir):
            print("⚠️ Persist directory not found.")
            return

        print(f"Contents of {persist_dir}: {os.listdir(persist_dir)}")
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="document_embeddings"
        )

        collection = db._collection
        count = collection.count()
        print(f"🧠 Collection has {count} documents.")
        if count > 0:
            results = collection.get(limit=3)
            for i, (doc_id, content, meta) in enumerate(zip(
                results.get('ids', []),
                results.get('documents', []),
                results.get('metadatas', [])
            )):
                print(f"📄 Doc {i+1}: {doc_id}")
                print(f"    Content: {content[:200]}...")
                print(f"    Metadata: {meta}")
                print("-" * 40)

    except Exception as e:
        import traceback
        traceback.print_exc()

def test_query_system():
    print("🔧 Testing query system...")
    connected, msg = check_ollama_connection()
    print("✓" if connected else "✗", msg)

    debug_database_contents()

    try:
        docs = get_relevant_documents("test query", k=1)
        if docs:
            print(f"✓ Vector DB working, {len(docs)} doc(s) returned.")
        else:
            print("⚠️ No documents found.")
    except Exception as e:
        print(f"✗ Error testing database: {e}")

if __name__ == "__main__":
    test_query_system()
