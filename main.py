"""
Phishing Email Detection System - Main Entry Point
==================================================
A machine learning system to detect phishing emails

Usage:
    python main.py                    # Full pipeline
    python main.py --samples 5000     # Custom sample count
    python main.py --interactive      # Interactive mode only
    python main.py --model rf         # Use specific model
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

from colorama import Fore, Style, init
init(autoreset=True)

from data_generator import EmailDatasetGenerator
from feature_extractor import EmailFeatureExtractor
from model_trainer import PhishingModelTrainer
from evaluator import ModelEvaluator
from predictor import PhishingPredictor


def print_banner():
    """Display startup banner"""
    print(f"""
{Fore.CYAN}{Style.BRIGHT}
╔══════════════════════════════════════════════════════════════╗
║          🎣 PHISHING EMAIL DETECTION SYSTEM v1.0            ║
║          Machine Learning Security Tool                      ║
╠══════════════════════════════════════════════════════════════╣
║  Algorithm : Random Forest + Gradient Boosting + SVM        ║
║  Features  : URL Analysis + Text Analysis + Header Check    ║
║  Output    : Phishing / Legitimate Classification           ║
╚══════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}""")


def run_sample_predictions(predictor: PhishingPredictor):
    """Run predictions on sample emails"""
    print(f"\n{Fore.CYAN}{'═' * 65}")
    print("SAMPLE EMAIL PREDICTIONS")
    print(f"{'═' * 65}{Style.RESET_ALL}")

    test_emails = [
        # Clear phishing
        {
            "sender": "security@paypa1-alert.xyz",
            "subject": "URGENT: Your account has been suspended!",
            "body": (
                "Dear Customer, URGENT ACTION REQUIRED! "
                "Your PayPal account has been suspended due to suspicious activity. "
                "Click here immediately to verify your identity or your account "
                "will be permanently deleted within 24 hours. "
                "Enter your credit card details and social security number. "
                "Act NOW before it's too late!"
            ),
            "url": "http://paypa1-secure-verify.xyz/login",
        },
        # Legitimate email
        {
            "sender": "support@amazon.com",
            "subject": "Your order has been shipped",
            "body": (
                "Hi John, your order #123-456-789 has been shipped and will "
                "arrive within 3-5 business days. You can track your package "
                "using the link below. Thank you for shopping with us. "
                "If you have questions, contact support@amazon.com"
            ),
            "url": "https://www.amazon.com/orders/track/123456",
        },
        # Lottery scam
        {
            "sender": "winner@prize-center.tk",
            "subject": "Congratulations! You WON $1,000,000!",
            "body": (
                "CONGRATULATIONS! You have been randomly selected as our "
                "monthly lottery winner. You have WON $1,000,000 USD! "
                "To claim your prize immediately, click the link and provide "
                "your full name, bank account number, and routing number. "
                "This offer expires in 24 hours! Claim NOW!"
            ),
            "url": "http://claim-prize-now.ga/winner",
        },
        # Legitimate meeting email
        {
            "sender": "manager@company.com",
            "subject": "Team meeting scheduled for Friday",
            "body": (
                "Hi Team, I wanted to remind everyone about our quarterly "
                "review meeting scheduled for Friday at 2 PM. Please review "
                "the attached agenda and come prepared with your updates. "
                "The meeting link has been sent to your calendar. "
                "Let me know if you have any scheduling conflicts."
            ),
            "url": "",
        },
        # Suspicious but borderline
        {
            "sender": "noreply@google-security-update.net",
            "subject": "Security alert: New sign-in to your account",
            "body": (
                "We detected a new sign-in to your Google Account. "
                "If this was you, you can ignore this email. "
                "If you didn't sign in recently, click here to secure "
                "your account and change your password immediately."
            ),
            "url": "http://google-security-update.net/secure",
        },
    ]

    for i, email in enumerate(test_emails, 1):
        print(f"\n{Fore.WHITE}── Sample #{i} ──────────────────────────────────{Style.RESET_ALL}")
        result = predictor.predict(email)
        predictor.display_prediction(email, result)


def main():
    """Main execution pipeline"""
    parser = argparse.ArgumentParser(
        description="Phishing Email Detection System"
    )
    parser.add_argument(
        "--samples", type=int, default=2000,
        help="Number of email samples to generate (default: 2000)"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode after training"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip visualization plots"
    )
    parser.add_argument(
        "--save-model", action="store_true",
        help="Save trained model to disk"
    )

    args = parser.parse_args()

    print_banner()

    # ═══════════════════════════════════════════
    # STEP 1: Generate Dataset
    # ═══════════════════════════════════════════
    print(f"\n{Fore.CYAN}{'═'*65}")
    print("STEP 1: DATA GENERATION")
    print(f"{'═'*65}{Style.RESET_ALL}")

    generator = EmailDatasetGenerator(seed=42)
    df = generator.generate_dataset(total_samples=args.samples)

    # ═══════════════════════════════════════════
    # STEP 2: Feature Extraction
    # ═══════════════════════════════════════════
    print(f"\n{Fore.CYAN}{'═'*65}")
    print("STEP 2: FEATURE EXTRACTION")
    print(f"{'═'*65}{Style.RESET_ALL}")

    extractor = EmailFeatureExtractor()
    features_df = extractor.extract_all_features(df)
    labels = df["label"]

    print(f"\n{Fore.WHITE}Feature Preview (first 5 rows):{Style.RESET_ALL}")
    print(features_df.head().to_string(max_cols=8))

    # ═══════════════════════════════════════════
    # STEP 3: Model Training
    # ═══════════════════════════════════════════
    print(f"\n{Fore.CYAN}{'═'*65}")
    print("STEP 3: MODEL TRAINING")
    print(f"{'═'*65}{Style.RESET_ALL}")

    trainer = PhishingModelTrainer(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = trainer.prepare_data(features_df, labels)
    model_results = trainer.train_all_models()

    # ═══════════════════════════════════════════
    # STEP 4: Model Evaluation
    # ═══════════════════════════════════════════
    print(f"\n{Fore.CYAN}{'═'*65}")
    print("STEP 4: MODEL EVALUATION")
    print(f"{'═'*65}{Style.RESET_ALL}")

    evaluator = ModelEvaluator(
        model=trainer.best_model,
        X_test=X_test,
        y_test=y_test,
        model_name=trainer.best_model_name,
    )

    metrics = evaluator.print_detailed_report()

    # Feature importance
    importance_df = trainer.get_feature_importance(extractor.get_feature_names())
    if importance_df is not None:
        print(f"\n{Fore.CYAN}── TOP 10 MOST IMPORTANT FEATURES ──{Style.RESET_ALL}")
        print(importance_df.head(10).to_string(index=False))

    # ═══════════════════════════════════════════
    # STEP 5: Visualizations
    # ═══════════════════════════════════════════
    if not args.no_plot:
        print(f"\n{Fore.CYAN}{'═'*65}")
        print("STEP 5: GENERATING VISUALIZATIONS")
        print(f"{'═'*65}{Style.RESET_ALL}")

        evaluator.plot_all_visualizations(
            feature_importance_df=importance_df,
            model_results=model_results,
            save_path="phishing_detection_dashboard.png",
        )

    # ═══════════════════════════════════════════
    # STEP 6: Sample Predictions
    # ═══════════════════════════════════════════
    print(f"\n{Fore.CYAN}{'═'*65}")
    print("STEP 6: SAMPLE PREDICTIONS")
    print(f"{'═'*65}{Style.RESET_ALL}")

    predictor = PhishingPredictor(
        model=trainer.best_model,
        scaler=trainer.scaler,
        feature_extractor=extractor,
    )

    run_sample_predictions(predictor)

    # ═══════════════════════════════════════════
    # STEP 7: Save Model (optional)
    # ═══════════════════════════════════════════
    if args.save_model:
        trainer.save_model("phishing_model.pkl")

    # ═══════════════════════════════════════════
    # STEP 8: Interactive Mode (optional)
    # ═══════════════════════════════════════════
    if args.interactive:
        predictor.interactive_mode()

    # ── Final Summary ─────────────────────────
    print(f"\n{Fore.GREEN}{'═'*65}")
    print("✅ PHISHING DETECTION SYSTEM - COMPLETE")
    print(f"{'═'*65}{Style.RESET_ALL}")
    print(f"  Best Model  : {trainer.best_model_name}")
    print(f"  Accuracy    : {metrics['accuracy']*100:.2f}%")
    print(f"  F1-Score    : {metrics['f1_score']*100:.2f}%")
    print(f"  Precision   : {metrics['precision']*100:.2f}%")
    print(f"  Recall      : {metrics['recall']*100:.2f}%")
    print(f"\n  Dashboard saved: phishing_detection_dashboard.png")
    print(f"{Fore.GREEN}{'═'*65}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()