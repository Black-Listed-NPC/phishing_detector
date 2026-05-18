"""
Real-time Email Prediction Module
Allows testing individual emails against the trained model
"""

import numpy as np
from colorama import Fore, Style
from feature_extractor import EmailFeatureExtractor


class PhishingPredictor:
    """Real-time phishing prediction for individual emails"""

    RISK_LEVELS = {
        (0.0, 0.2):  ("✅ VERY SAFE",    Fore.GREEN + Style.BRIGHT,  "This email appears completely legitimate."),
        (0.2, 0.4):  ("✅ SAFE",         Fore.GREEN,                  "This email appears legitimate."),
        (0.4, 0.6):  ("⚠️  SUSPICIOUS",  Fore.YELLOW,                 "This email has some suspicious characteristics."),
        (0.6, 0.8):  ("🚨 LIKELY PHISH", Fore.RED,                    "This email is likely a phishing attempt!"),
        (0.8, 1.01): ("🚨 PHISHING",     Fore.RED + Style.BRIGHT,     "DANGER! This is almost certainly a phishing email!"),
    }

    def __init__(self, model, scaler, feature_extractor: EmailFeatureExtractor):
        self.model = model
        self.scaler = scaler
        self.extractor = feature_extractor

    def predict(self, email: dict) -> dict:
        """
        Predict if an email is phishing

        Args:
            email: Dictionary with sender, subject, body, url

        Returns:
            Prediction result dictionary
        """
        # Extract features
        features = self.extractor.extract_single_email(email)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Get prediction and probability
        prediction = self.model.predict(features_scaled)[0]
        probability = (
            self.model.predict_proba(features_scaled)[0][1]
            if hasattr(self.model, "predict_proba") else float(prediction)
        )

        # Determine risk level
        risk_label, risk_color, risk_message = self._get_risk_level(probability)

        return {
            "prediction": "PHISHING" if prediction == 1 else "LEGITIMATE",
            "is_phishing": bool(prediction),
            "confidence": probability,
            "risk_level": risk_label,
            "risk_color": risk_color,
            "risk_message": risk_message,
        }

    def _get_risk_level(self, probability: float) -> tuple:
        """Get risk level based on probability"""
        for (low, high), (label, color, message) in self.RISK_LEVELS.items():
            if low <= probability < high:
                return label, color, message
        return "UNKNOWN", Fore.WHITE, "Unable to determine risk level."

    def display_prediction(self, email: dict, result: dict):
        """
        Display formatted prediction result

        Args:
            email: Input email dictionary
            result: Prediction result dictionary
        """
        print(f"\n{Fore.CYAN}{'═' * 65}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'EMAIL ANALYSIS RESULT':^65}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'═' * 65}{Style.RESET_ALL}")

        # Email details
        print(f"\n{Fore.WHITE}📧 Email Details:{Style.RESET_ALL}")
        print(f"  From   : {email.get('sender', 'N/A')[:60]}")
        print(f"  Subject: {email.get('subject', 'N/A')[:60]}")
        print(f"  URL    : {email.get('url', 'None')[:60]}")

        # Result
        color = result["risk_color"]
        print(f"\n{Fore.WHITE}🔍 Analysis Result:{Style.RESET_ALL}")
        print(f"  Status    : {color}{result['risk_level']}{Style.RESET_ALL}")
        print(f"  Verdict   : {color}{result['prediction']}{Style.RESET_ALL}")
        print(f"  Confidence: {self._confidence_bar(result['confidence'])}")
        print(f"  Message   : {result['risk_message']}")

        # Warning box for phishing
        if result["is_phishing"]:
            print(f"\n  {Fore.RED}┌─────────────────────────────────────────────┐")
            print(f"  │  ⚠️  DO NOT click any links in this email!   │")
            print(f"  │  ⚠️  Do not provide any personal information! │")
            print(f"  │  ⚠️  Report this email to your IT department! │")
            print(f"  └─────────────────────────────────────────────┘{Style.RESET_ALL}")

        print(f"{Fore.CYAN}{'═' * 65}{Style.RESET_ALL}")

    def _confidence_bar(self, probability: float) -> str:
        """Create visual confidence bar"""
        bar_length = 30
        filled = int(probability * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        color = (Fore.GREEN if probability < 0.4 else
                Fore.YELLOW if probability < 0.6 else Fore.RED)

        return f"{color}[{bar}]{Style.RESET_ALL} {probability*100:.1f}%"

    def interactive_mode(self):
        """Run interactive prediction mode"""
        print(f"\n{Fore.CYAN}{'═' * 65}")
        print("INTERACTIVE PHISHING DETECTOR")
        print(f"{'═' * 65}{Style.RESET_ALL}")
        print("Enter email details for real-time phishing detection.")
        print(f"Type '{Fore.YELLOW}quit{Style.RESET_ALL}' to exit.\n")

        while True:
            try:
                print(f"\n{Fore.CYAN}── New Email Analysis ──{Style.RESET_ALL}")
                sender = input("Sender email: ").strip()

                if sender.lower() == "quit":
                    print(f"\n{Fore.YELLOW}Exiting detector...{Style.RESET_ALL}")
                    break

                subject = input("Subject: ").strip()
                body = input("Body (paste text): ").strip()
                url = input("URL (press Enter if none): ").strip()

                email = {
                    "sender": sender,
                    "subject": subject,
                    "body": body,
                    "url": url,
                }

                result = self.predict(email)
                self.display_prediction(email, result)

            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}Detector stopped.{Style.RESET_ALL}")
                break