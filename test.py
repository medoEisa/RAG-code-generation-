# # from gc.math import add 
# # add(5, 7)




# # from typing import List


# # def has_close_elements(numbers: List[float], threshold: float) -> bool:
# # """ Check if in given list of numbers, are any two numbers closer to each other than
# # given threshold.
# # >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
# # False
# # >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
# # True
# # """


# # def double_the_difference(lst):
# # '''
# # Given a list of numbers, return the sum of squares of the numbers
# # in the list that are odd. Ignore numbers that are negative or not integers.

# # double_the_difference([1, 3, 2, 0]) == 1 + 9 + 0 + 0 = 10
# # double_the_difference([-1, -2, 0]) == 0
# # double_the_difference([9, -2]) == 81
# # double_the_difference([0]) == 0

# # If the input list is empty, return 0.
# # '''


# #Write a function that finds all pairs of numbers in a list that sum to a target value

# # "Generate a function that removes duplicates from a list while preserving order."

# import streamlit as st
# from generate import generate_code
# from embedding import retrieve_similar, init_chroma
# from evaluate import evaluate_retrieval

# st.set_page_config(page_title="RAG Code Generation Demo", page_icon="💡", layout="wide")

# st.title(" RAG-based Code Generation System")
# st.markdown(
#     "This app demonstrates **Retrieval-Augmented Code Generation** using the HumanEval dataset "
#     "+ a causal LM to highlight the power of retrieval."
# )

# # settings
# st.sidebar.header(" Settings")
# rerank = st.sidebar.checkbox("Use Reranking (Cross-Encoder)", value=False)
# top_k = st.sidebar.slider("Top-K Retrieved Examples", 1, 10, 3)
# show_eval = st.sidebar.checkbox("Run Retrieval Evaluation", value=False)

# # prompt
# prompt = st.text_area(
#     "Enter your function prompt:",
#     placeholder="Example: def fibonacci(n):",
#     height=150
# )

# if st.button(" Generate Code"):
#     if not prompt.strip():
#         st.warning("Please enter a valid code prompt.")
#     else:
#         st.info(" Retrieving similar examples from ChromaDB...")
#         try:
#             docs, metas = retrieve_similar("humaneval", prompt, top_k=top_k, rerank=rerank)
#         except Exception as e:
#             st.error(f"Error retrieving examples: {e}")
#             st.stop()

#         st.subheader("📚 Retrieved Examples")
#         if not docs:
#             st.warning("No similar examples found in the database.")
#         else:
#             for i, (doc, meta) in enumerate(zip(docs, metas)):
#                 with st.expander(f"Example {i+1} — Task ID: {meta.get('task_id', 'N/A') if isinstance(meta, dict) else 'N/A'}"):
#                     st.code(doc.strip(), language="python")
#                     if isinstance(meta, dict) and "canonical_solution" in meta:
#                         st.caption(f"Solution available in metadata (truncated).")

#         st.info(" Generating code ...")
#         try:
#             generated_code = generate_code(prompt, top_k=top_k, rerank=rerank)
#             st.subheader("🧠 Generated Code")
#             st.code(generated_code, language="python")
#         except Exception as e:
#             st.error(f"Error generating code: {e}")

# if show_eval:
#     st.markdown("---")
#     st.subheader(" Retrieval Quality Evaluation")
#     with st.spinner("Running retrieval evaluation..."):
#         try:
#             recall, mrr = evaluate_retrieval(rerank=rerank)
#             st.success(" Evaluation complete.")
#             st.write(f"**Recall@{top_k}:** {recall:.3f}")
#             st.write(f"**MRR:** {mrr:.3f}")
#         except Exception as e:
#             st.error(f"Error during evaluation: {e}")

# st.markdown("---")
# st.caption("Built with Salesforce/codegen + ChromaDB + Streamlit")
