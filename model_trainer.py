"""
Model Training for Phishing Email Detection
Trains multiple ML models and selects the best one
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from colorama import Fore, Style
from tabulate import tabulate


class PhishingModelTrainer:
    """Trains and compares multiple ML models for phishing detection"""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Define all models to compare
        self.models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state,
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                C=1.0,
            ),
            "Naive Bayes": GaussianNB(),
            "K-Nearest Neighbors": KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
            ),
            "Support Vector Machine": SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                random_state=random_state,
            ),
        }

        self.model_results = {}

    def prepare_data(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series
    ) -> tuple:
        """
        Prepare and split data for training

        Args:
            features_df: Feature DataFrame
            labels: Target labels

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"\n{Fore.CYAN}[*] Preparing training data...{Style.RESET_ALL}")

        # Handle missing values
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(features_df)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            labels.values,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )

        # Scale features
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        print(f"{Fore.GREEN}[✓] Data prepared:{Style.RESET_ALL}")
        print(f"    Training samples : {len(X_train)}")
        print(f"    Testing samples  : {len(X_test)}")
        print(f"    Feature count    : {X_train.shape[1]}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_all_models(self) -> dict:
        """
        Train all models and compare performance

        Returns:
            Dictionary of model results
        """
        print(f"\n{Fore.CYAN}[*] Training {len(self.models)} models...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'─' * 60}{Style.RESET_ALL}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        results = []

        for name, model in self.models.items():
            print(f"\n  {Fore.WHITE}Training: {name}...{Style.RESET_ALL}")

            try:
                # Cross-validation
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=cv, scoring="accuracy", n_jobs=-1
                )

                # Fit on full training data
                model.fit(self.X_train, self.y_train)

                # Test accuracy
                test_accuracy = model.score(self.X_test, self.y_test)

                self.model_results[name] = {
                    "model": model,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "test_accuracy": test_accuracy,
                    "cv_scores": cv_scores,
                }

                color = Fore.GREEN if test_accuracy > 0.90 else (
                    Fore.YELLOW if test_accuracy > 0.80 else Fore.RED
                )

                print(
                    f"  {color}✓ {name}: "
                    f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f} | "
                    f"Test={test_accuracy:.4f}{Style.RESET_ALL}"
                )

                results.append({
                    "Model": name,
                    "CV Mean": f"{cv_scores.mean():.4f}",
                    "CV Std": f"±{cv_scores.std():.4f}",
                    "Test Accuracy": f"{test_accuracy:.4f}",
                    "Status": "✓ Excellent" if test_accuracy > 0.90 else (
                        "● Good" if test_accuracy > 0.80 else "✗ Poor"
                    ),
                })

            except Exception as e:
                print(f"  {Fore.RED}✗ {name} failed: {e}{Style.RESET_ALL}")

        # Display comparison table
        print(f"\n{Fore.CYAN}{'═' * 70}")
        print("MODEL COMPARISON RESULTS")
        print(f"{'═' * 70}{Style.RESET_ALL}")
        print(tabulate(results, headers="keys", tablefmt="grid"))

        # Select best model
        self._select_best_model()

        return self.model_results

    def _select_best_model(self):
        """Select the best performing model"""
        if not self.model_results:
            return

        best_name = max(
            self.model_results,
            key=lambda x: self.model_results[x]["test_accuracy"]
        )

        self.best_model = self.model_results[best_name]["model"]
        self.best_model_name = best_name

        best_acc = self.model_results[best_name]["test_accuracy"]
        print(f"\n{Fore.GREEN}[✓] Best Model Selected: {best_name}")
        print(f"    Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%){Style.RESET_ALL}")

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importances from best model

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature importances sorted by importance
        """
        if not hasattr(self.best_model, "feature_importances_"):
            return None

        importances = self.best_model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False)

        return importance_df

    def save_model(self, filepath: str = "phishing_model.pkl"):
        """
        Save trained model to disk

        Args:
            filepath: Path to save model
        """
        if self.best_model is None:
            print(f"{Fore.RED}[✗] No model to save{Style.RESET_ALL}")
            return

        model_data = {
            "model": self.best_model,
            "scaler": self.scaler,
            "model_name": self.best_model_name,
        }

        joblib.dump(model_data, filepath)
        print(f"{Fore.GREEN}[✓] Model saved: {filepath}{Style.RESET_ALL}")

    def load_model(self, filepath: str = "phishing_model.pkl"):
        """
        Load trained model from disk

        Args:
            filepath: Path to model file
        """
        if not os.path.exists(filepath):
            print(f"{Fore.RED}[✗] Model file not found: {filepath}{Style.RESET_ALL}")
            return False

        model_data = joblib.load(filepath)
        self.best_model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.best_model_name = model_data["model_name"]

        print(f"{Fore.GREEN}[✓] Model loaded: {self.best_model_name}{Style.RESET_ALL}")
        return True