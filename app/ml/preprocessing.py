import pandas as pd


DATASET_PATH = "data/raw/filtered_dataset.xlsx"

OUTPUT_PATH = "data/processed/cleaned_dataset.csv"


def preprocess_dataset():

    # Load dataset
    df = pd.read_excel(DATASET_PATH)

    print("\nDataset Loaded Successfully")
    print(f"Original Shape: {df.shape}")

    # Remove duplicate URLs
    df = df.drop_duplicates(subset=["URL"])

    print(f"\nAfter Removing Duplicates: {df.shape}")

    # Remove missing values
    df = df.dropna()

    print(f"\nAfter Removing Missing Values: {df.shape}")

    # Clean URLs
    df["URL"] = df["URL"].str.lower()

    df["URL"] = df["URL"].str.strip()

    print("\nURLs cleaned successfully")

    # Encode labels explicitly
    # Assuming:
    # 1 = Legitimate
    # 0 = Phishing

    df["label"] = df["label"].astype(int)

    print("\nLabel Encoding Completed")

    # Save processed dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nProcessed dataset saved to:\n{OUTPUT_PATH}")

    # Show preview
    print("\nFirst 5 Rows:")
    print(df.head())


if __name__ == "__main__":
    preprocess_dataset()