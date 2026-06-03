import os
import tempfile
import requests
import joblib
import pandas as pd

from app.agents.url_agent import (
    parse_url,
    check_suspicious_tld,
    check_ip_address,
    check_suspicious_keywords,
    check_excessive_subdomains,
)

from app.agents.html_agent import analyze_html

from app.ml.feature_extraction import extract_features


# =====================================
# LOAD MODELS
# =====================================

url_model = joblib.load(
    "data/processed/logistic_model.pkl"
)

html_model = joblib.load(
    "data/processed/html_model.pkl"
)


# =====================================
# VERDICT GENERATOR
# =====================================

def generate_final_verdict(score):

    if score >= 7:
        return "HIGH RISK PHISHING"

    elif score >= 4:
        return "MEDIUM RISK"

    else:
        return "LOW RISK / LEGITIMATE"


# =====================================
# DOWNLOAD HTML
# =====================================

def fetch_html(url):

    response = requests.get(
        url,
        timeout=10
    )

    return response.text


# =====================================
# MAIN ANALYSIS
# =====================================

def analyze_website(url):

    final_score = 0

    reasons = []

    # =================================
    # URL AGENT
    # =================================

    parsed = parse_url(url)

    tld_result = check_suspicious_tld(
        parsed["domain"]
    )

    ip_result = check_ip_address(
        parsed["domain"]
    )

    keyword_result = check_suspicious_keywords(
        url
    )

    subdomain_result = check_excessive_subdomains(
        parsed["domain"]
    )

    url_risk = (
        tld_result["risk_score"]
        + ip_result["risk_score"]
        + keyword_result["risk_score"]
        + subdomain_result["risk_score"]
    )

    final_score += url_risk

    reasons.append(
        f"URL Agent Risk Score: {url_risk}"
    )

    # =================================
    # URL ML MODEL
    # =================================

    url_features = extract_features(url)

    url_df = pd.DataFrame([url_features])

    url_prediction = url_model.predict(
        url_df
    )[0]

    if url_prediction == 0:

        final_score += 3

        reasons.append(
            "URL ML Model predicts phishing"
        )

    # =================================
    # FETCH HTML
    # =================================

    try:

        html_content = fetch_html(url)

        # Save temporary HTML
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".html",
            mode="w",
            encoding="utf-8"
        ) as temp_file:

            temp_file.write(html_content)

            temp_html_path = temp_file.name

        # =============================
        # HTML AGENT
        # =============================

        html_results = analyze_html(
            temp_html_path
        )

        html_risk = html_results[
            "risk_score"
        ]

        final_score += html_risk

        reasons.extend(
            html_results["reasons"]
        )

        # =============================
        # HTML ML MODEL
        # =============================

        html_features = {

            "num_forms":
                html_results["num_forms"],

            "num_password_inputs":
                html_results[
                    "num_password_inputs"
                ],

            "num_iframes":
                html_results[
                    "num_iframes"
                ],

            "num_scripts":
                html_results[
                    "num_scripts"
                ],

            "num_external_scripts":
                html_results[
                    "num_external_scripts"
                ],

            "num_links":
                html_results[
                    "num_links"
                ],
        }

        html_df = pd.DataFrame(
            [html_features]
        )

        html_prediction = html_model.predict(
            html_df
        )[0]

        if html_prediction == 0:

            final_score += 3

            reasons.append(
                "HTML ML Model predicts phishing"
            )

        # Cleanup temp file
        os.remove(temp_html_path)

    except Exception as e:

        reasons.append(
            f"HTML Analysis Failed: {str(e)}"
        )

        html_prediction = -1

    # =================================
    # FINAL VERDICT
    # =================================

    final_verdict = generate_final_verdict(
        final_score
    )

    return {

        "url": url,

        "final_score": final_score,

        "final_verdict": final_verdict,

        "reasons": reasons,

        "url_ml_prediction":
            int(url_prediction),

        "html_ml_prediction":
            int(html_prediction),
    }


# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    test_url = (
        "https://uktech.ac.in/en"
    )

    results = analyze_website(
        test_url
    )

    print("\nFINAL RESULTS\n")

    for key, value in results.items():

        print(f"{key}: {value}")