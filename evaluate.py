import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from embedding import collection, retrieve_similar, load
from generate import build_rag_prompt
from nltk.translate.bleu_score import sentence_bleu

def load_codegen_model(model_name="Salesforce/codegen-350M-multi"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def generate_solution(query, top_k=1, max_examples=1):

    docs, metas = retrieve_similar(collection, query, top_k=top_k, rerank=True)

    tokenizer, model = load_codegen_model()
    prompt = build_rag_prompt(query, docs, max_examples=max_examples)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
        )

    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    gen_part = generated_code[len(prompt):].strip()
    return metas[:max_examples], gen_part

def evaluate_example(query, canonical_solution, top_k=1, max_examples=1):
   
    metas, generated = generate_solution(query, top_k=top_k, max_examples=max_examples)

    exact_match = int(generated.strip() == canonical_solution.strip())

    reference = [canonical_solution.strip().split()]
    candidate = generated.strip().split()
    bleu = sentence_bleu(reference, candidate)

    print("\n====================")
    print("Query:", query)
    print("\nRetrieved Examples:")
    for i, meta in enumerate(metas):
        print(f"  Task ID: {meta['task_id']}")
        print(f"  Canonical Solution (truncated): {meta['solution'][:200]} ...")

    print("\nGenerated Solution:\n", generated)
    print(f"\nEvaluation Metrics:\n  Exact Match: {exact_match}\n  BLEU Score: {bleu:.4f}")
    return exact_match, bleu

def evaluate_dataset(sample_size=5):
  
    data = load()
    data_sample = data.sample(sample_size, random_state=42)

    total_em = 0
    total_bleu = 0
    for idx, row in data_sample.iterrows():
        em, bleu = evaluate_example(row['prompt'], row['canonical_solution'])
        total_em += em
        total_bleu += bleu

    print("\n====================")
    print(f"Average Exact Match: {total_em/sample_size:.2f}")
    print(f"Average BLEU Score: {total_bleu/sample_size:.4f}")

if __name__ == "__main__":
    evaluate_dataset(sample_size=5)
