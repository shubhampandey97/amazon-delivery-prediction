import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


input_path = BASE_DIR / "data" / "processed" / "cleaned_data.csv"
output_path = BASE_DIR / "data" / "processed" / "final_data.csv"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def create_features(df):
    # Distance feature
    df['distance_km'] = haversine(
        df['Store_Latitude'], df['Store_Longitude'],
        df['Drop_Latitude'], df['Drop_Longitude']
    )

    # Convert datetime
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])

    # Extract time features
    df['day'] = df['Order_Date'].dt.day
    df['month'] = df['Order_Date'].dt.month
    df['weekday'] = df['Order_Date'].dt.weekday
    
    # Order preparation time
    df['Order_Time'] = pd.to_datetime(df['Order_Time'])
    df['Pickup_Time'] = pd.to_datetime(df['Pickup_Time'])
    df['prep_time'] = (df['Pickup_Time'] - df['Order_Time']).dt.total_seconds() / 60

    return df

if __name__ == "__main__":
    if not input_path.exists():
        raise FileNotFoundError(f"❌ Cleaned data not found at {input_path}")

    df = pd.read_csv(input_path)
    df = create_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Final data saved at: {output_path}")