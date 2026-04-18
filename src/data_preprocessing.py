import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "raw" / "amazon_delivery.csv"
output_path = BASE_DIR / "data" / "processed" / "cleaned_data.csv"

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna()

    # Standardize categorical columns
    df['Weather'] = df['Weather'].str.strip().str.lower()
    df['Traffic'] = df['Traffic'].str.strip().str.lower()

    return df

if __name__ == "__main__":
    df = load_data(input_path)
    df = clean_data(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Cleaned data saved at: {output_path}")