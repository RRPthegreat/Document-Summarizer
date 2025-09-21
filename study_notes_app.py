import streamlit as st
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from collections import Counter
from textwrap import wrap

# -------------------------------
# 1️⃣ Load Hugging Face Summarizer
# -------------------------------
model_name = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype="float32",
    low_cpu_mem_usage=False
)

summarizer_pipeline = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=-1,
    max_length=300,
    min_length=100,
    do_sample=False
)

llm = HuggingFacePipeline(pipeline=summarizer_pipeline)

# -------------------------------
# 2️⃣ Prompt Template for Short Summary
# -------------------------------
prompt = PromptTemplate.from_template(
    "Summarize the following text into a concise study note. "
    "The summary should be 6-7 lines long (about 100-120 words) and highlight key points only:\n\n{text}\n\nSummary:"
)
chain = prompt | llm

# -------------------------------
# 3️⃣ Helper Functions
# -------------------------------
def chunk_text_by_tokens(text, tokenizer, max_tokens=500):
    """Split text into safe chunks based on tokenizer limits."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        tokens = tokenizer(" ".join(current_chunk), return_tensors="pt")["input_ids"]
        if tokens.shape[1] > max_tokens:
            # remove last word and finalize chunk
            current_chunk.pop()
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_text(text):
    """Summarize text safely using token-based chunking."""
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=500)
    summaries = []

    for chunk in chunks:
        try:
            partial_summary = chain.invoke({"text": chunk})
            if isinstance(partial_summary, dict) and "text" in partial_summary:
                summaries.append(partial_summary["text"])
            else:
                summaries.append(str(partial_summary))
        except Exception as e:
            summaries.append(f"[Error summarizing chunk: {e}]")

    final_input = " ".join(summaries)
    if final_input.strip():
        try:
            final_summary = chain.invoke({"text": final_input})
            if isinstance(final_summary, dict) and "text" in final_summary:
                wrapped_summary = "\n".join(wrap(final_summary["text"], width=100))
            else:
                wrapped_summary = "\n".join(wrap(str(final_summary), width=100))
        except Exception as e:
            wrapped_summary = f"[Error in final summary: {e}]"
    else:
        wrapped_summary = "[No text to summarize]"

    return wrapped_summary

def suggest_citations_api(text, max_results=5):
    try:
        words = [w.lower() for w in text.split() if len(w) > 4]
        keywords = " ".join([w for w, _ in Counter(words).most_common(5)])

        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={keywords}&limit={max_results}&fields=title,authors,year,url"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            citations = []
            for paper in data.get("data", []):
                title = paper.get("title", "Untitled")
                authors = ", ".join([a["name"] for a in paper.get("authors", [])[:2]]) or "Unknown"
                year = paper.get("year", "n.d.")
                link = paper.get("url", "")
                citations.append(f"[{title}]({link}) - {authors} ({year})")
            return citations if citations else ["⚠️ No citations found."]
        else:
            return [f"⚠️ API Error: {response.status_code}"]

    except Exception as e:
        return [f"⚠️ Could not fetch citations: {e}"]

# -------------------------------
# 4️⃣ Theme Switcher (Dark / Light)
# -------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

if st.button("🌙 Toggle Theme"):
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

if st.session_state.theme == "dark":
    st.markdown("""
        <style>
        body { background-color: #1e1e1e; color: white; }
        .stTextArea textarea { background-color: #333333 !important; color: white !important; }
        .stButton button { background-color: #444444; color: white; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body { background-color: white; color: black; }
        .stTextArea textarea { background-color: #f9f9f9 !important; color: black !important; }
        .stButton button { background-color: #e0e0e0; color: black; }
        </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 5️⃣ Streamlit UI
# -------------------------------
st.title("📘 SummarIQ – Paper Summarizer ")
st.write("Paste abstract or upload PDF/DOCX, then click Summarize to get key points with suggested citations.!")

uploaded_file = st.file_uploader("Upload PDF/DOCX file", type=["pdf", "docx"])
user_input = ""
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        user_input = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        from docx import Document
        doc = Document(uploaded_file)
        user_input = " ".join([para.text for para in doc.paragraphs if para.text])
else:
    user_input = st.text_area("Or enter text here:", height=200)

if st.button("Summarize"):
    if not user_input.strip():
        st.warning("Please enter or upload some text first!")
    else:
        with st.spinner("Generating concise summary..."):
            summary = summarize_text(user_input)
            st.subheader("🔹 Summary of Text")
            st.write(summary)

            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append(summary)

            with st.expander("📜 Previous Summaries"):
                for i, h in enumerate(st.session_state.history):
                    st.write(f"Summary {i+1}: {h}")

            st.download_button(
                label="📥 Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )

            with st.expander("🔗 Suggested References"):
                for c in suggest_citations_api(user_input):
                    st.markdown(c, unsafe_allow_html=True)

