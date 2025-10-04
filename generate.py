import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from embedding import collection, retrieve_similar

def load_codegen_model(model_name="Salesforce/codegen-350M-multi"):
   
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def build_rag_prompt(query, docs, max_examples=1):
  
    context = "\n\n".join(f"# Example {i+1}\n{docs[i]}" for i in range(min(max_examples, len(docs))))
    final_prompt = (
        f"### Task:\n{query}\n\n"
        f"### Reference Examples:\n{context}\n\n"
        f"### Solution (Python function only i don't need any ccomments):\n"
    )
    return final_prompt

def generate_code(query, top_k=1, max_examples=1):
   
    print(f"\nRetrieving related examples...")
    docs, metas = retrieve_similar(collection, query, top_k=top_k, rerank=True)

    print("\nRetrieved Examples:")
    for i, meta in enumerate(metas[:max_examples]):
        print(f"\n--- Example {i+1} ---")
        print("Task ID:", meta["task_id"])
        print("Original Solution (truncated):", meta["solution"][:500], "...")

    tokenizer, model = load_codegen_model()
    prompt = build_rag_prompt(query, docs, max_examples=max_examples)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256*2,
            temperature=0.7,
            do_sample=True,
            top_p=0.90,
        )

    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    gen_part = generated_code[len(prompt):].strip()

    print("\n Generated Solution:\n")
    print(gen_part)

    return metas[:max_examples], gen_part

if __name__ == "__main__":
    user_query = input("Query: ")
    generate_code(user_query)
