import pandas as pd
from urllib.parse import urlparse
import re


INPUT_PATH = "data/processed/cleaned_dataset.csv"

OUTPUT_PATH = "data/processed/featured_dataset.csv"


SUSPICIOUS_TLDS = {
    "xyz", "tk", "ml", "ga", "cf", "gq",
    "buzz", "top", "club", "work",
    "info", "online", "site", "icu",
}


SUSPICIOUS_KEYWORDS = {
    "login",
    "verify",
    "secure",
    "account",
    "bank",
    "update",
    "signin",
    "payment",
    "wallet",
}


def has_ip_address(domain):

    ip_pattern = r"^\d{1,3}(\.\d{1,3}){3}$"

    return int(
        re.match(ip_pattern, domain) is not None
    )


def count_suspicious_keywords(url):

    count = 0

    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in url:
            count += 1

    return count


def extract_features(url):

    parsed = urlparse(url)

    domain = parsed.hostname or ""

    parts = domain.split(".")

    tld = parts[-1] if len(parts) > 0 else ""

    features = {

        "url_length": len(url),

        "num_dots": url.count("."),

        "num_hyphens": url.count("-"),

        "num_slashes": url.count("/"),

        "has_ip": has_ip_address(domain),

        "has_at_symbol": int("@" in url),

        "suspicious_tld": int(
            tld in SUSPICIOUS_TLDS
        ),

        "keyword_count": count_suspicious_keywords(url),

        "subdomain_count": max(len(parts) - 2, 0),
    }

    return features


def process_dataset():

    df = pd.read_csv(INPUT_PATH)

    feature_rows = []

    for _, row in df.iterrows():

        url = row["URL"]

        features = extract_features(url)

        features["label"] = row["label"]

        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)

    feature_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature Extraction Completed")

    print(f"\nFeature Dataset Saved To:\n{OUTPUT_PATH}")

    print("\nFirst 5 Rows:")

    print(feature_df.head())


if __name__ == "__main__":
    process_dataset()