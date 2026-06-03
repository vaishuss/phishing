from bs4 import BeautifulSoup


def analyze_html(html_path):

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

    # Extract elements
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

    # ======================================
    # Risk Scoring
    # ======================================

    risk_score = 0

    reasons = []

    # Password fields
    if len(password_inputs) > 0:

        risk_score += 2

        reasons.append(
            "Password input field detected"
        )

    # Excessive iframes
    if len(iframes) >= 3:

        risk_score += 2

        reasons.append(
            "Multiple iframes detected"
        )

    # Too many external scripts
    if external_scripts >= 20:

        risk_score += 1

        reasons.append(
            "Large number of external scripts"
        )

    # Too many forms
    if len(forms) >= 3:

        risk_score += 1

        reasons.append(
            "Multiple forms detected"
        )

    # Excessive links
    if len(links) >= 100:

        risk_score += 1

        reasons.append(
            "Large number of links detected"
        )

    # Final verdict
    verdict = (
        "Suspicious"
        if risk_score >= 3
        else "Legitimate"
    )

    results = {

        "num_forms": len(forms),

        "num_password_inputs": len(
            password_inputs
        ),

        "num_iframes": len(iframes),

        "num_scripts": len(scripts),

        "num_external_scripts": external_scripts,

        "num_links": len(links),

        "risk_score": risk_score,

        "reasons": reasons,

        "verdict": verdict,
    }

    return results


if __name__ == "__main__":

    sample_file = (
        "data/raw/html_processed_files/1.html"
    )

    analysis = analyze_html(sample_file)

    print("\nHTML ANALYSIS RESULTS\n")

    for key, value in analysis.items():

        print(f"{key}: {value}")