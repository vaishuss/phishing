import os
import pandas as pd

from bs4 import BeautifulSoup


HTML_FOLDER = "data/raw/html_processed_files"

DATASET_PATH = "data/raw/filtered_dataset.xlsx"

OUTPUT_PATH = "data/processed/html_features.csv"


def extract_html_features(html_path):

    with open(
        html_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        html_content = file.read()

    soup = BeautifulSoup(
        html_content,
        "html.parser"
    )

    # Extract HTML elements
    forms = soup.find_all("form")

    password_inputs = soup.find_all(
        "input",
        {"type": "password"}
    )

    iframes = soup.find_all("iframe")

    scripts = soup.find_all("script")

    links = soup.find_all("a")

    # Count external scripts
    external_scripts = 0

    for script in scripts:

        if script.get("src"):
            external_scripts += 1

    # Feature dictionary
    return {

        "num_forms": len(forms),

        "num_password_inputs": len(
            password_inputs
        ),

        "num_iframes": len(iframes),

        "num_scripts": len(scripts),

        "num_external_scripts": external_scripts,

        "num_links": len(links),
    }


def process_html_dataset():

    # Load original dataset
    df = pd.read_excel(DATASET_PATH)

    rows = []

    html_files = os.listdir(HTML_FOLDER)

    print(f"\nTotal HTML Files Found: {len(html_files)}")

    for file_name in html_files:

        # Ignore non-html files
        if not file_name.endswith(".html"):
            continue

        try:

            # Extract ID from filename
            file_id = int(
                file_name.replace(".html", "")
            )

            # Match dataset row
            matching_row = df[
                df["ID"] == file_id
            ]

            if matching_row.empty:
                continue

            label = matching_row.iloc[0]["label"]

            html_path = os.path.join(
                HTML_FOLDER,
                file_name
            )

            # Extract features
            features = extract_html_features(
                html_path
            )

            # Add metadata
            features["file_id"] = file_id

            features["label"] = label

            rows.append(features)

        except Exception as e:

            print(f"\nError Processing: {file_name}")

            print(e)

    # Create dataframe
    feature_df = pd.DataFrame(rows)

    # Save dataset
    feature_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nHTML Feature Extraction Completed")

    print(
        f"\nDataset Saved To:\n{OUTPUT_PATH}"
    )

    print("\nFirst 5 Rows:\n")

    print(feature_df.head())


if __name__ == "__main__":
    process_html_dataset()