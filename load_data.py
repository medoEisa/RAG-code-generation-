from datasets import load_dataset
import pandas as pd

def load():
    
    data = load_dataset("openai/openai_humaneval")
    data = data["test"].to_pandas()
    data = pd.DataFrame((data[["task_id","prompt", "canonical_solution"]]))

    return data

