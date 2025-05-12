import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

# Load the smaller, faster model
@st.cache_resource
def load_fast_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_fast_model()

# Streamlit UI
st.title("⚡ Fast Research Paper Summarizer (DistilBART)")

uploaded_file = st.file_uploader("📁 Upload a Research Paper (PDF)", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = "".join(page.extract_text() for page in reader.pages if page.extract_text())

    st.subheader("📖 Extracted Text Preview")
    st.write(text[:1000] + "..." if len(text) > 1000 else text)

    # Split text into chunks to fit model input size
    chunk_size = 1000
    chunk_overlap = 100
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])

    if st.button("🧠 Summarize"):
        st.info("Summarizing with DistilBART (Fast Model)...")
        summaries = []
        for i, chunk in enumerate(chunks):
            st.write(f"⏳ Processing chunk {i + 1} of {len(chunks)}...")
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
            summaries.append(summary)

        final_summary = "\n\n".join(summaries)
        st.subheader("📌 Final Summary")
        st.write(final_summary)

