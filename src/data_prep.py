import os
import pandas as pd
from datasets import load_dataset

RAW_DATASET_NAME = "CShorten/ML-ArXiv-Papers"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLEANED_CSV_PATH = os.path.join(DATA_DIR, "cleaned_arxiv_papers.csv")
MAX_PAPERS = 50_000

def load_raw_data() -> pd.DataFrame:
    print(f"Loading dataset '{RAW_DATASET_NAME}' from Hugging Face...")
    dataset = load_dataset(RAW_DATASET_NAME)
    df = dataset["train"].to_pandas()
    return df


def clean_data(df: pd.DataFrame, max_papers: int = MAX_PAPERS) -> pd.DataFrame:
    df = df[["title", "abstract"]].copy()
    df = df.head(max_papers)
    df = df.dropna(subset=["title", "abstract"]).reset_index(drop=True)
    df["paper_text"] = df["title"] + " " + df["abstract"]
    df["paper_text"] = df["paper_text"].str.replace("\n", " ", regex=False)
    df["paper_text"] = df["paper_text"].str.strip()

    return df


def save_cleaned_data(df: pd.DataFrame, path: str = CLEANED_CSV_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Cleaned dataframe saved to {path} ({len(df):,} papers)")


def get_cleaned_data(force_reload: bool = False) -> pd.DataFrame:
    if not force_reload and os.path.exists(CLEANED_CSV_PATH):
        print("Loading cached cleaned dataset...")
        return pd.read_csv(CLEANED_CSV_PATH)

    df = load_raw_data()
    df = clean_data(df)
    save_cleaned_data(df)
    return df

if __name__ == "__main__":
    get_cleaned_data(force_reload=True)
