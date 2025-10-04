import streamlit as st
from generate import generate_code
from embedding import retrieve_similar, collection

st.set_page_config(page_title="RAG Code Generation Demo", page_icon="💡", layout="wide")

st.title("RAG-based Code Generation System")
st.markdown(
    "This app demonstrates **Retrieval-Augmented Code Generation** using the HumanEval dataset "
    "and a causal LM to highlight the power of retrieval."
)

# Sidebar settings
st.sidebar.header("Settings")
rerank = st.sidebar.checkbox("Use Reranking (Cross-Encoder)", value=False)
top_k = st.sidebar.slider("Top-K Retrieved Examples", 1, 10, 3)
max_examples = st.sidebar.slider("Max Examples in Prompt", 1, 3, 1)

# Prompt input
prompt = st.text_area(
    "Enter your function prompt:",
    placeholder="Example: Write a Python function to reverse a string",
    height=150
)

if st.button("Generate Code"):
    if not prompt.strip():
        st.warning("Please enter a valid code prompt.")
    else:
        st.info("🔍 Retrieving similar examples from ChromaDB...")
        try:
            docs, metas = retrieve_similar(collection, prompt, top_k=top_k, rerank=rerank)
        except Exception as e:
            st.error(f"Error retrieving examples: {e}")
            st.stop()

        st.subheader("📚 Retrieved Examples")
        if not docs:
            st.warning("No similar examples found in the database.")
        else:
            for i, meta in enumerate(metas[:max_examples]):
                with st.expander(f"Example {i+1} — Task ID: {meta.get('task_id', 'N/A')}"):
                    st.code(docs[i].strip(), language="python")
                    if "solution" in meta:
                        st.caption(f"Canonical Solution (truncated): {meta['solution'][:200]} ...")

        st.info("🧠 Generating code ...")
        try:
            retrieved_metas, generated_code = generate_code(prompt, top_k=top_k, max_examples=max_examples)
            st.subheader("📝 Generated Code")
            st.code(generated_code, language="python")
        except Exception as e:
            st.error(f"Error generating code: {e}")

st.markdown("---")
st.caption("Built with Salesforce/codegen + ChromaDB + Streamlit")
