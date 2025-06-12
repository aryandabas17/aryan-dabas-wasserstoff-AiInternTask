import streamlit as st
from streamlit import markdown
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Fix for PyTorch/Streamlit compatibility issue
try:
    import torch
    torch.set_num_threads(1)
    # Set environment variable to avoid the torch.classes issue
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
except ImportError:
    pass

# Import your services
try:
    from backend.app.services import ingest, embed, query, summarize
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.error("Please make sure all dependencies are installed correctly.")
    st.stop()

# Initialize session state
if "docs" not in st.session_state:
    st.session_state.docs = None

if "theme" not in st.session_state:
    st.session_state.theme = None

if "history" not in st.session_state:
    st.session_state.history = []

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# Page configuration
st.set_page_config(
    page_title="Document Research Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Document Research Assistant")
st.markdown("Upload documents and ask questions about their content!")

# File upload section
uploaded_files = st.file_uploader(
    "Upload image(s) or PDF(s)",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    try:
        # Create directories
        os.makedirs("data/input_images", exist_ok=True)
        os.makedirs("data/text_outputs", exist_ok=True)
        
        # Save uploaded files
        for file in uploaded_files:
            file_path = os.path.join("data/input_images", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())

        st.success("✅ Files uploaded successfully.")
        
        # Process files
        with st.spinner("🔍 Extracting text from documents..."):
            ingest.process_files("data/input_images", "data/text_outputs")
        st.success("✅ Text extraction completed!")

        # Load and embed documents
        with st.spinner("📚 Loading and embedding documents..."):
            docs = embed.load_texts("data/text_outputs")
            if docs:
                vectorstore = embed.embed_documents(docs)
                if vectorstore:
                    st.session_state.docs = docs
                    st.session_state.vectorstore_ready = True
                    st.success("✅ Documents embedded successfully!")
                else:
                    st.error("❌ Failed to create embeddings")
            else:
                st.error("❌ No text extracted from documents")

    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")

# Theme summarization section
if st.session_state.docs and st.button("🔎 Summarize Theme"):
    try:
        with st.spinner("📝 Generating theme summary..."):
            full_text = "\n".join([doc.page_content for doc in st.session_state.docs])
            summary = summarize.get_theme_summary(full_text)
            st.session_state.theme = summary
        st.success("✅ Theme extracted!")
    except Exception as e:
        st.error(f"❌ Error generating summary: {str(e)}")

# Display theme summary if available
if st.session_state.theme:
    st.markdown("### 🧩 Theme Summary")
    st.info(st.session_state.theme)

# Only show Q&A section if documents are ready
if st.session_state.vectorstore_ready:
    st.divider()
    
    # Q&A Section
    st.markdown("### 💬 Ask a Question")
    query_text = st.text_input("🔍 Type your question about the documents")

    col1, col2 = st.columns([1, 2])
    with col1:
        ask_clicked = st.button("Ask", use_container_width=True)
    with col2:
        clear_clicked = st.button("🧹 Clear History", use_container_width=True)

    # Process question
    if ask_clicked and query_text.strip():
        try:
            with st.spinner("🤖 Searching for answer..."):
                answer = query.ask_question(query_text)
            st.session_state.history.append((query_text, answer))
            st.success("✅ Answer generated!")
        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")

    # Clear chat history
    if clear_clicked:
        st.session_state.history.clear()
        st.success("🧹 History cleared!")

    # Display Q&A history
    if st.session_state.history:
        st.markdown("### 📜 Q&A History")
        for i, (q, a) in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Q{i}: {q[:50]}..." if len(q) > 50 else f"Q{i}: {q}"):
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Answer:** {a}")

elif st.session_state.docs is not None:
    st.info("📚 Documents are being processed. Please wait...")
else:
    st.info("📤 Please upload some documents to get started!")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and local AI models")
