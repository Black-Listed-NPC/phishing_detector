"""
Feature Extraction for Phishing Email Detection
Extracts meaningful features from raw email data
"""

import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from colorama import Fore, Style


class EmailFeatureExtractor:
    """Extracts features from emails for ML classification"""

    # ── Suspicious keyword lists ─────────────────────────────────
    URGENT_WORDS = [
        "urgent", "immediately", "now", "today", "hours",
        "expire", "suspend", "terminate", "warning", "alert",
        "critical", "final", "last chance", "act now", "hurry",
        "deadline", "limited time", "asap", "right away",
    ]

    PHISHING_KEYWORDS = [
        "verify", "confirm", "update", "validate", "authenticate",
        "click here", "login", "sign in", "account suspended",
        "unusual activity", "unauthorized", "security breach",
        "password expired", "verify identity", "confirm account",
        "reactivate", "unlock", "restore access",
    ]

    MONEY_KEYWORDS = [
        "winner", "won", "prize", "lottery", "reward", "free",
        "gift", "congratulations", "selected", "claim",
        "million", "billion", "dollars", "cash", "money",
        "inheritance", "transfer", "bitcoin", "crypto",
    ]

    THREAT_WORDS = [
        "suspended", "terminated", "deleted", "closed", "blocked",
        "restricted", "limited", "disabled", "cancelled", "banned",
        "expired", "overdue", "penalty", "fine", "lawsuit",
    ]

    LEGITIMATE_KEYWORDS = [
        "meeting", "schedule", "invoice", "receipt", "order",
        "shipped", "delivered", "tracking", "appointment",
        "reminder", "update", "newsletter", "report",
        "quarterly", "annual", "weekly", "monthly",
    ]

    # ── Suspicious TLDs ─────────────────────────────────────────
    SUSPICIOUS_TLDS = [
        ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq",
        ".ru", ".cn", ".pw", ".top", ".click", ".download",
        ".win", ".loan", ".review", ".stream",
    ]

    LEGITIMATE_TLDS = [
        ".com", ".org", ".net", ".edu", ".gov",
        ".io", ".co", ".app", ".dev",
    ]

    def __init__(self):
        self.feature_names = []

    def extract_url_features(self, url: str) -> dict:
        """
        Extract features from URL

        Args:
            url: URL string to analyze

        Returns:
            Dictionary of URL features
        """
        features = {}

        if not url or url == "":
            return {
                "url_length": 0,
                "has_url": 0,
                "url_has_ip": 0,
                "url_has_suspicious_tld": 0,
                "url_has_legitimate_tld": 0,
                "url_subdomain_count": 0,
                "url_has_https": 0,
                "url_has_http_only": 0,
                "url_special_char_count": 0,
                "url_digit_count": 0,
                "url_has_shortener": 0,
                "url_has_at_symbol": 0,
                "url_has_double_slash": 0,
                "url_path_length": 0,
                "url_has_suspicious_keywords": 0,
                "url_has_numbers_in_domain": 0,
                "url_hyphen_count": 0,
                "url_dot_count": 0,
            }

        features["has_url"] = 1
        features["url_length"] = len(url)

        try:
            parsed = urlparse(url)
            domain = parsed.netloc or ""
            path = parsed.path or ""

            # Protocol checks
            features["url_has_https"] = 1 if url.startswith("https://") else 0
            features["url_has_http_only"] = 1 if url.startswith("http://") else 0

            # IP address in URL (suspicious)
            ip_pattern = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
            features["url_has_ip"] = 1 if re.search(ip_pattern, domain) else 0

            # TLD analysis
            features["url_has_suspicious_tld"] = int(
                any(url.lower().endswith(tld) or f"{tld}/" in url.lower()
                    for tld in self.SUSPICIOUS_TLDS)
            )
            features["url_has_legitimate_tld"] = int(
                any(domain.lower().endswith(tld)
                    for tld in self.LEGITIMATE_TLDS)
            )

            # Domain analysis
            domain_parts = domain.split(".")
            features["url_subdomain_count"] = max(0, len(domain_parts) - 2)
            features["url_has_numbers_in_domain"] = int(bool(re.search(r"\d", domain)))
            features["url_hyphen_count"] = domain.count("-")
            features["url_dot_count"] = url.count(".")

            # Special characters
            special_chars = re.findall(r"[!@#$%^&*()_+=\[\]{};':\"\\|,<>?`~]", url)
            features["url_special_char_count"] = len(special_chars)
            features["url_digit_count"] = sum(c.isdigit() for c in url)

            # Suspicious patterns
            shorteners = ["bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly",
                         "is.gd", "buff.ly", "adf.ly", "tiny.cc"]
            features["url_has_shortener"] = int(
                any(s in url.lower() for s in shorteners)
            )
            features["url_has_at_symbol"] = int("@" in url)
            features["url_has_double_slash"] = int("//" in path)

            # Path length
            features["url_path_length"] = len(path)

            # Suspicious keywords in URL
            suspicious_url_words = ["login", "verify", "secure", "update",
                                   "confirm", "account", "banking", "prize"]
            features["url_has_suspicious_keywords"] = int(
                any(word in url.lower() for word in suspicious_url_words)
            )

        except Exception:
            # Return safe defaults on parse error
            for key in ["url_has_ip", "url_has_suspicious_tld", "url_has_legitimate_tld",
                       "url_subdomain_count", "url_has_https", "url_has_http_only",
                       "url_special_char_count", "url_digit_count", "url_has_shortener",
                       "url_has_at_symbol", "url_has_double_slash", "url_path_length",
                       "url_has_suspicious_keywords", "url_has_numbers_in_domain",
                       "url_hyphen_count", "url_dot_count"]:
                features.setdefault(key, 0)

        return features

    def extract_sender_features(self, sender: str) -> dict:
        """
        Extract features from email sender address

        Args:
            sender: Email sender string

        Returns:
            Dictionary of sender features
        """
        sender = sender.lower() if sender else ""

        # Extract domain from email
        domain = ""
        if "@" in sender:
            domain = sender.split("@")[1]

        return {
            "sender_domain_length": len(domain),
            "sender_has_numbers": int(bool(re.search(r"\d", domain))),
            "sender_has_suspicious_tld": int(
                any(domain.endswith(tld.strip("."))
                    for tld in self.SUSPICIOUS_TLDS)
            ),
            "sender_has_legitimate_domain": int(
                any(legit in domain for legit in [
                    "google", "microsoft", "amazon", "apple",
                    "paypal", "netflix", "github", "linkedin",
                    "gmail", "outlook", "yahoo",
                ])
            ),
            "sender_has_hyphen": int("-" in domain),
            "sender_domain_has_numbers": int(bool(re.search(r"\d", domain))),
            "sender_has_special_chars": int(
                bool(re.search(r"[^a-z0-9@._-]", sender))
            ),
            "sender_subdomain_count": max(0, domain.count(".") - 1),
            "sender_has_zero_substitution": int(
                bool(re.search(r"[a-z]0[a-z]|0[a-z]|[a-z]0", domain))
            ),  # e.g., "amaz0n"
            "sender_length": len(sender),
        }

    def extract_subject_features(self, subject: str) -> dict:
        """
        Extract features from email subject line

        Args:
            subject: Email subject string

        Returns:
            Dictionary of subject features
        """
        subject_lower = subject.lower() if subject else ""

        return {
            "subject_length": len(subject),
            "subject_word_count": len(subject.split()),
            "subject_urgent_word_count": sum(
                1 for word in self.URGENT_WORDS if word in subject_lower
            ),
            "subject_has_exclamation": int("!" in subject),
            "subject_exclamation_count": subject.count("!"),
            "subject_has_question": int("?" in subject),
            "subject_is_uppercase": int(subject.isupper()),
            "subject_uppercase_ratio": (
                sum(1 for c in subject if c.isupper()) / max(len(subject), 1)
            ),
            "subject_has_money_keyword": int(
                any(word in subject_lower for word in self.MONEY_KEYWORDS)
            ),
            "subject_has_threat_word": int(
                any(word in subject_lower for word in self.THREAT_WORDS)
            ),
            "subject_has_phishing_keyword": int(
                any(word in subject_lower for word in self.PHISHING_KEYWORDS)
            ),
            "subject_digit_count": sum(c.isdigit() for c in subject),
            "subject_has_free": int("free" in subject_lower),
            "subject_has_winner": int(
                any(w in subject_lower for w in ["winner", "won", "win"])
            ),
            "subject_has_urgent": int(
                any(w in subject_lower for w in ["urgent", "immediately", "asap"])
            ),
        }

    def extract_body_features(self, body: str) -> dict:
        """
        Extract features from email body text

        Args:
            body: Email body string

        Returns:
            Dictionary of body features
        """
        body_lower = body.lower() if body else ""
        words = body_lower.split()
        sentences = re.split(r"[.!?]", body)

        return {
            # Basic stats
            "body_length": len(body),
            "body_word_count": len(words),
            "body_sentence_count": len(sentences),
            "body_avg_word_length": (
                np.mean([len(w) for w in words]) if words else 0
            ),

            # Keyword counts
            "body_urgent_count": sum(
                body_lower.count(word) for word in self.URGENT_WORDS
            ),
            "body_phishing_keyword_count": sum(
                1 for word in self.PHISHING_KEYWORDS if word in body_lower
            ),
            "body_money_keyword_count": sum(
                1 for word in self.MONEY_KEYWORDS if word in body_lower
            ),
            "body_threat_count": sum(
                1 for word in self.THREAT_WORDS if word in body_lower
            ),
            "body_legit_keyword_count": sum(
                1 for word in self.LEGITIMATE_KEYWORDS if word in body_lower
            ),

            # Pattern detection
            "body_url_count": len(re.findall(r"https?://\S+", body)),
            "body_email_count": len(re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", body)),
            "body_phone_count": len(re.findall(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", body)),
            "body_exclamation_count": body.count("!"),
            "body_question_count": body.count("?"),

            # Suspicious patterns
            "body_has_click_here": int("click here" in body_lower),
            "body_has_dear_customer": int(
                any(phrase in body_lower for phrase in
                    ["dear customer", "dear user", "dear member", "valued customer"])
            ),
            "body_has_personal_info_request": int(
                any(phrase in body_lower for phrase in [
                    "credit card", "social security", "ssn", "bank account",
                    "routing number", "password", "pin number",
                ])
            ),
            "body_has_time_pressure": int(
                any(phrase in body_lower for phrase in [
                    "24 hours", "48 hours", "within hours", "expires today",
                    "limited time", "act now", "immediately",
                ])
            ),
            "body_has_lottery": int(
                any(phrase in body_lower for phrase in [
                    "lottery", "sweepstakes", "prize", "winner", "selected"
                ])
            ),

            # Text quality indicators
            "body_uppercase_ratio": (
                sum(1 for c in body if c.isupper()) / max(len(body), 1)
            ),
            "body_digit_ratio": (
                sum(1 for c in body if c.isdigit()) / max(len(body), 1)
            ),
            "body_special_char_count": len(re.findall(r"[!@#$%^&*]", body)),
        }

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from email DataFrame

        Args:
            df: DataFrame with email data

        Returns:
            DataFrame with extracted features
        """
        print(f"\n{Fore.CYAN}[*] Extracting email features...{Style.RESET_ALL}")

        all_features = []

        for idx, row in df.iterrows():
            features = {}

            # Extract from each component
            features.update(self.extract_url_features(row.get("url", "")))
            features.update(self.extract_sender_features(row.get("sender", "")))
            features.update(self.extract_subject_features(row.get("subject", "")))
            features.update(self.extract_body_features(row.get("body", "")))

            all_features.append(features)

        features_df = pd.DataFrame(all_features)
        self.feature_names = list(features_df.columns)

        print(f"{Fore.GREEN}[✓] Features extracted:{Style.RESET_ALL}")
        print(f"    Total features : {len(self.feature_names)}")
        print(f"    Samples        : {len(features_df)}")
        print(f"    Feature groups : URL({18}), Sender({10}), Subject({15}), Body({24})")

        return features_df

    def get_feature_names(self) -> list:
        """Return list of feature names"""
        return self.feature_names

    def extract_single_email(self, email: dict) -> np.ndarray:
        """
        Extract features from a single email for prediction

        Args:
            email: Dictionary with email components

        Returns:
            Feature array for model prediction
        """
        features = {}
        features.update(self.extract_url_features(email.get("url", "")))
        features.update(self.extract_sender_features(email.get("sender", "")))
        features.update(self.extract_subject_features(email.get("subject", "")))
        features.update(self.extract_body_features(email.get("body", "")))

        return np.array(list(features.values())).reshape(1, -1)