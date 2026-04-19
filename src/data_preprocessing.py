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

    df = remove_outliers(df, 'Delivery_Time')

    return df

def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[col] >= lower) & (df[col] <= upper)]

if __name__ == "__main__":
    df = load_data(input_path)
    df = clean_data(df)
    df = remove_outliers(df, 'Delivery_Time')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Cleaned data saved at: {output_path}")