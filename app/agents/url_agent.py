from urllib.parse import urlparse
import re


SUSPICIOUS_TLDS = {
    "xyz", "tk", "ml", "ga", "cf", "gq",
    "buzz", "top", "club", "work",
    "info", "online", "site", "icu",
    "monster", "surf", "rest", "fit",
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


def parse_url(url: str) -> dict:
    """Break URL into components."""

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)

    return {
        "scheme": parsed.scheme,
        "domain": parsed.hostname or "",
        "port": parsed.port,
        "path": parsed.path,
        "query": parsed.query,
        "fragment": parsed.fragment,
    }


def check_suspicious_tld(domain: str) -> dict:
    """Check suspicious TLD."""

    parts = domain.split(".")
    tld = parts[-1].lower() if parts else ""

    is_suspicious = tld in SUSPICIOUS_TLDS

    return {
        "check": "suspicious_tld",
        "tld": tld,
        "is_suspicious": is_suspicious,
        "risk_score": 1.0 if is_suspicious else 0.0,
        "reason": (
            f"TLD '.{tld}' is commonly used in phishing"
            if is_suspicious
            else "TLD appears normal"
        ),
    }


def check_ip_address(domain: str) -> dict:
    """Detect raw IP address."""

    ip_pattern = r"^\d{1,3}(\.\d{1,3}){3}$"

    is_ip = re.match(ip_pattern, domain) is not None

    return {
        "check": "ip_address",
        "is_suspicious": is_ip,
        "risk_score": 2.0 if is_ip else 0.0,
        "reason": (
            "Domain uses raw IP address"
            if is_ip
            else "Domain uses normal hostname"
        ),
    }


def check_suspicious_keywords(url: str) -> dict:
    """Check phishing-related keywords."""

    found_keywords = []

    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in url.lower():
            found_keywords.append(keyword)

    is_suspicious = len(found_keywords) > 0

    return {
        "check": "suspicious_keywords",
        "keywords_found": found_keywords,
        "is_suspicious": is_suspicious,
        "risk_score": len(found_keywords) * 0.5,
        "reason": (
            f"Suspicious keywords detected: {', '.join(found_keywords)}"
            if is_suspicious
            else "No suspicious keywords detected"
        ),
    }


def check_excessive_subdomains(domain: str) -> dict:
    """Detect excessive subdomains."""

    parts = domain.split(".")

    subdomain_count = max(len(parts) - 2, 0)

    is_suspicious = subdomain_count >= 2

    return {
        "check": "subdomain_count",
        "subdomain_count": subdomain_count,
        "is_suspicious": is_suspicious,
        "risk_score": 1.0 if is_suspicious else 0.0,
        "reason": (
            "Too many subdomains detected"
            if is_suspicious
            else "Subdomain count appears normal"
        ),
    }