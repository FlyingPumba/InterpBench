from huggingface_hub import hf_hub_download
import pandas as pd

def get_working_cases() -> list[str]:
    file = hf_hub_download(
      "cybershiptrooper/InterpBench",
      filename="benchmark_cases_metadata.csv",
    )
    df = pd.read_csv(file)
    working_cases = df["case_id"]
    return working_cases.tolist()