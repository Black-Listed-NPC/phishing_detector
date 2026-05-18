"""
Dataset Generator for Phishing Email Detection
Creates realistic phishing and legitimate email samples
"""

import pandas as pd
import numpy as np
import random
import re
from colorama import Fore, Style


class EmailDatasetGenerator:
    """Generates realistic email datasets for training"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        # ── Legitimate email components ──────────────────────────
        self.legit_senders = [
            "support@amazon.com", "noreply@google.com",
            "info@microsoft.com", "hello@github.com",
            "newsletter@linkedin.com", "team@slack.com",
            "billing@netflix.com", "no-reply@apple.com",
            "service@paypal.com", "contact@spotify.com",
            "updates@twitter.com", "news@medium.com",
            "hr@company.com", "boss@workplace.com",
            "friend@gmail.com", "colleague@outlook.com",
        ]

        self.legit_subjects = [
            "Your monthly statement is ready",
            "Meeting scheduled for tomorrow",
            "Project update: Q4 Report",
            "Welcome to our service",
            "Your order has been shipped",
            "Invoice #12345 for your records",
            "Weekly team standup notes",
            "Happy Birthday! Here's a gift",
            "Your subscription renewal",
            "New comment on your post",
            "Reminder: Doctor appointment tomorrow",
            "Re: Lunch plans for Friday",
            "Quarterly performance review",
            "Your package is out for delivery",
            "New message from John Smith",
            "Team building event next week",
        ]

        self.legit_bodies = [
            "Hi {name}, please find attached your monthly account statement. "
            "If you have any questions, contact our support team at support@company.com.",

            "Dear {name}, your order #{order} has been shipped and will arrive "
            "within 3-5 business days. Track your package at our website.",

            "Hello {name}, this is a reminder about our team meeting scheduled "
            "for tomorrow at 10 AM. Please review the agenda before joining.",

            "Hi {name}, thank you for your purchase. Your invoice is attached. "
            "Payment is due within 30 days. Contact billing@company.com for help.",

            "Dear {name}, we wanted to let you know that your subscription "
            "will automatically renew on the 15th. No action is needed.",

            "Hi there, just checking in to see if you're available for lunch "
            "on Friday? Let me know what works for you.",

            "Dear {name}, your quarterly performance review is scheduled. "
            "Please prepare a self-assessment before the meeting.",

            "Hello {name}, we noticed you haven't logged in recently. "
            "We miss you! Here are some updates from your network.",

            "Hi {name}, the project deadline has been extended to next Friday. "
            "Please update your tasks in the project management tool.",

            "Dear Team, please join us for the annual company picnic next Saturday. "
            "RSVP by Wednesday. Food and drinks will be provided.",
        ]

        self.legit_urls = [
            "https://www.amazon.com/orders/track",
            "https://accounts.google.com/login",
            "https://www.microsoft.com/support",
            "https://github.com/dashboard",
            "https://www.linkedin.com/notifications",
            "https://www.paypal.com/account",
            "https://www.netflix.com/account",
            "https://support.apple.com/help",
            "https://www.spotify.com/account",
            "https://www.company.com/invoice/12345",
        ]

        # ── Phishing email components ────────────────────────────
        self.phishing_senders = [
            "support@amaz0n-security.com",
            "noreply@g00gle-verify.net",
            "security@paypa1-alert.com",
            "admin@micros0ft-update.xyz",
            "verify@apple-id-confirm.tk",
            "alert@netflix-billing.ga",
            "team@dropbox-secure.ml",
            "no-reply@bankofamerica-verify.com",
            "security@chase-alert.net",
            "update@ebay-suspended.com",
            "help@amazon.security-update.ru",
            "noreply@google.account-verify.tk",
            "support@paypal.secure-login.xyz",
            "admin@microsoft365-update.ml",
            "verify@icloud-suspended.net",
        ]

        self.phishing_subjects = [
            "URGENT: Your account has been suspended!",
            "Action Required: Verify your account immediately",
            "Security Alert: Suspicious login detected",
            "Your account will be DELETED in 24 hours",
            "Congratulations! You've won $1,000,000!",
            "IMMEDIATE ACTION REQUIRED: Update payment info",
            "Your PayPal account is limited - Act NOW",
            "FINAL WARNING: Account suspension notice",
            "You have unclaimed reward points - Claim NOW",
            "Unusual activity detected on your account",
            "ALERT: Your password was compromised",
            "Verify your identity or lose access",
            "FREE iPhone 15 - You've been selected!",
            "Your bank account needs immediate verification",
            "Important: Confirm your email or account closes",
            "Last chance: Claim your prize before it expires",
        ]

        self.phishing_bodies = [
            "Dear Valued Customer, We have detected suspicious activity on your "
            "account. CLICK HERE IMMEDIATELY to verify your identity or your "
            "account will be permanently suspended within 24 hours. "
            "Login Now: {url} Your account is at risk!",

            "URGENT NOTICE: Your {service} account has been compromised. "
            "We need you to verify your personal information immediately. "
            "Failure to do so will result in account termination. "
            "Click here to verify: {url} Act NOW before it's too late!",

            "Congratulations! You have been randomly selected to receive a "
            "$1,000 gift card. Claim your prize immediately at {url} "
            "This offer expires in 2 hours! Provide your details to claim.",

            "Dear Customer, Your payment information needs to be updated. "
            "Your account will be suspended if you don't update within 24 hours. "
            "Update now: {url} Enter your credit card details to continue.",

            "SECURITY ALERT: We detected a login from an unknown device. "
            "If this wasn't you, click here immediately: {url} "
            "Your account password and personal data may be compromised.",

            "You have WON our monthly lottery! Prize: $500,000 USD. "
            "To claim your prize, provide your banking details at: {url} "
            "This is NOT spam. You were specially selected. Act fast!",

            "Your account password will expire in 24 hours. "
            "Update your password immediately to avoid losing access: {url} "
            "Enter your current password and create a new one. URGENT!",

            "Final Notice: We couldn't process your payment. "
            "Update your billing information NOW or lose your subscription: {url} "
            "Required: Credit card number, CVV, expiry date.",

            "Dear Winner, Your email has won £850,000 in our online lottery. "
            "Send your: Full name, address, phone number, bank details to: {url} "
            "Keep this confidential. Claim within 48 hours.",

            "ACCOUNT ALERT: Unusual login attempt detected from Russia. "
            "Secure your account immediately: {url} "
            "Enter your username, password, and security questions NOW.",
        ]

        self.phishing_urls = [
            "http://amaz0n-secure-login.xyz/verify",
            "http://paypa1-account-update.tk/login",
            "http://g00gle-security-check.ml/auth",
            "http://apple-id-suspended.ga/verify",
            "http://netflix-billing-update.ru/account",
            "http://secure-bankofamerica.xyz/login",
            "http://microsoft-update-required.net/fix",
            "http://prize-claim-center.tk/winner",
            "http://account-verify-now.ml/confirm",
            "http://secure-login-required.xyz/auth",
            "http://ebay-account-suspended.ga/restore",
            "http://chase-security-alert.net/verify",
            "http://192.168.1.1/phishing-page",
            "http://bit.ly/3xPhish1ng",
            "http://tinyurl.com/fake-bank-login",
        ]

    def generate_email(self, is_phishing: bool) -> dict:
        """
        Generate a single email sample

        Args:
            is_phishing: Whether to generate phishing or legitimate email

        Returns:
            Dictionary containing email features
        """
        if is_phishing:
            sender = random.choice(self.phishing_senders)
            subject = random.choice(self.phishing_subjects)
            url = random.choice(self.phishing_urls)
            body_template = random.choice(self.phishing_bodies)
            body = body_template.format(
                url=url,
                service=random.choice(["PayPal", "Amazon", "Google", "Apple", "Netflix"]),
                name=random.choice(["Customer", "User", "Account Holder", "Valued Member"]),
            )
            label = 1  # Phishing
        else:
            sender = random.choice(self.legit_senders)
            subject = random.choice(self.legit_subjects)
            url = random.choice(self.legit_urls) if random.random() > 0.3 else ""
            body_template = random.choice(self.legit_bodies)
            body = body_template.format(
                name=random.choice(["John", "Sarah", "Team", "Everyone"]),
                order=random.randint(100000, 999999),
            )
            label = 0  # Legitimate

        return {
            "sender": sender,
            "subject": subject,
            "body": body,
            "url": url,
            "label": label,
        }

    def generate_dataset(self, total_samples: int = 2000) -> pd.DataFrame:
        """
        Generate complete dataset with balanced classes

        Args:
            total_samples: Total number of email samples

        Returns:
            DataFrame with email samples
        """
        print(f"\n{Fore.CYAN}[*] Generating email dataset...{Style.RESET_ALL}")

        half = total_samples // 2
        emails = []

        # Generate phishing emails
        for _ in range(half):
            emails.append(self.generate_email(is_phishing=True))

        # Generate legitimate emails
        for _ in range(half):
            emails.append(self.generate_email(is_phishing=False))

        # Shuffle dataset
        random.shuffle(emails)
        df = pd.DataFrame(emails)

        # Stats
        phishing_count = df["label"].sum()
        legit_count = len(df) - phishing_count

        print(f"{Fore.GREEN}[✓] Dataset generated:{Style.RESET_ALL}")
        print(f"    Total emails  : {len(df)}")
        print(f"    Phishing      : {phishing_count} ({phishing_count/len(df)*100:.1f}%)")
        print(f"    Legitimate    : {legit_count} ({legit_count/len(df)*100:.1f}%)")

        return df