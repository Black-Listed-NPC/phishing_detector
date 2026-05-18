"""
Model Evaluation for Phishing Email Detection
Generates detailed metrics, confusion matrix, and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
)
from colorama import Fore, Style
from tabulate import tabulate


class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""

    def __init__(self, model, X_test, y_test, model_name: str = "Model"):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.y_pred = model.predict(X_test)
        self.y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba") else None
        )

    def calculate_metrics(self) -> dict:
        """
        Calculate all evaluation metrics

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "precision": precision_score(self.y_test, self.y_pred, zero_division=0),
            "recall": recall_score(self.y_test, self.y_pred, zero_division=0),
            "f1_score": f1_score(self.y_test, self.y_pred, zero_division=0),
        }

        if self.y_prob is not None:
            fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
            metrics["auc_roc"] = auc(fpr, tpr)
        else:
            metrics["auc_roc"] = None

        return metrics

    def print_detailed_report(self):
        """Print comprehensive evaluation report to console"""
        metrics = self.calculate_metrics()
        cm = confusion_matrix(self.y_test, self.y_pred)

        print(f"\n{Fore.CYAN}{'═' * 65}")
        print(f"{'PHISHING DETECTION - MODEL EVALUATION REPORT':^65}")
        print(f"{'═' * 65}{Style.RESET_ALL}")

        # Model info
        print(f"\n{Fore.WHITE}Model: {Fore.CYAN}{self.model_name}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Test Samples: {len(self.y_test)}{Style.RESET_ALL}")

        # ── Performance Metrics ──────────────────────────────────
        print(f"\n{Fore.CYAN}── PERFORMANCE METRICS {'─' * 40}{Style.RESET_ALL}")

        metrics_table = [
            ["Accuracy",  f"{metrics['accuracy']:.4f}",  f"{metrics['accuracy']*100:.2f}%",  self._grade(metrics['accuracy'])],
            ["Precision", f"{metrics['precision']:.4f}", f"{metrics['precision']*100:.2f}%", self._grade(metrics['precision'])],
            ["Recall",    f"{metrics['recall']:.4f}",    f"{metrics['recall']*100:.2f}%",    self._grade(metrics['recall'])],
            ["F1-Score",  f"{metrics['f1_score']:.4f}",  f"{metrics['f1_score']*100:.2f}%",  self._grade(metrics['f1_score'])],
        ]

        if metrics["auc_roc"]:
            metrics_table.append([
                "AUC-ROC", f"{metrics['auc_roc']:.4f}",
                f"{metrics['auc_roc']*100:.2f}%", self._grade(metrics['auc_roc'])
            ])

        print(tabulate(
            metrics_table,
            headers=["Metric", "Score", "Percentage", "Grade"],
            tablefmt="grid"
        ))

        # ── Confusion Matrix ─────────────────────────────────────
        print(f"\n{Fore.CYAN}── CONFUSION MATRIX {'─' * 43}{Style.RESET_ALL}")

        tn, fp, fn, tp = cm.ravel()
        total = len(self.y_test)

        cm_display = [
            ["", f"{Fore.WHITE}Predicted SAFE{Style.RESET_ALL}", f"{Fore.WHITE}Predicted PHISH{Style.RESET_ALL}"],
            [
                f"{Fore.WHITE}Actual SAFE{Style.RESET_ALL}",
                f"{Fore.GREEN}TN={tn} ✓{Style.RESET_ALL}",
                f"{Fore.RED}FP={fp} ✗{Style.RESET_ALL}",
            ],
            [
                f"{Fore.WHITE}Actual PHISH{Style.RESET_ALL}",
                f"{Fore.RED}FN={fn} ✗{Style.RESET_ALL}",
                f"{Fore.GREEN}TP={tp} ✓{Style.RESET_ALL}",
            ],
        ]

        print(tabulate(cm_display, tablefmt="grid"))

        # Confusion matrix explanation
        print(f"\n  {Fore.GREEN}TN (True Negative) : {tn:4d} - Legitimate emails correctly identified{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}TP (True Positive) : {tp:4d} - Phishing emails correctly detected{Style.RESET_ALL}")
        print(f"  {Fore.RED}FP (False Positive): {fp:4d} - Legit emails wrongly flagged as phishing{Style.RESET_ALL}")
        print(f"  {Fore.RED}FN (False Negative): {fn:4d} - Phishing emails that slipped through{Style.RESET_ALL}")

        # ── Detailed Classification Report ───────────────────────
        print(f"\n{Fore.CYAN}── CLASSIFICATION REPORT {'─' * 37}{Style.RESET_ALL}")
        report = classification_report(
            self.y_test,
            self.y_pred,
            target_names=["Legitimate", "Phishing"],
            digits=4,
        )
        print(report)

        # ── Risk Analysis ─────────────────────────────────────────
        print(f"\n{Fore.CYAN}── RISK ANALYSIS {'─' * 45}{Style.RESET_ALL}")
        phishing_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"  Phishing Detection Rate : {phishing_detection_rate*100:.2f}% "
              f"({'✓ Excellent' if phishing_detection_rate > 0.95 else '⚠ Needs Improvement'})")
        print(f"  False Alarm Rate        : {false_alarm_rate*100:.2f}% "
              f"({'✓ Good' if false_alarm_rate < 0.05 else '⚠ Too High'})")
        print(f"  Missed Phishing Emails  : {fn} ({fn/total*100:.1f}%)")

        return metrics

    def _grade(self, score: float) -> str:
        """Grade a metric score"""
        if score >= 0.95:
            return f"{Fore.GREEN}A+ Excellent{Style.RESET_ALL}"
        elif score >= 0.90:
            return f"{Fore.GREEN}A  Great{Style.RESET_ALL}"
        elif score >= 0.85:
            return f"{Fore.YELLOW}B  Good{Style.RESET_ALL}"
        elif score >= 0.80:
            return f"{Fore.YELLOW}C  Fair{Style.RESET_ALL}"
        else:
            return f"{Fore.RED}D  Needs Work{Style.RESET_ALL}"

    def plot_all_visualizations(
        self,
        feature_importance_df: pd.DataFrame = None,
        model_results: dict = None,
        save_path: str = "phishing_detection_report.png"
    ):
        """
        Generate comprehensive visualization dashboard

        Args:
            feature_importance_df: DataFrame of feature importances
            model_results: Dictionary of all model results
            save_path: Path to save the figure
        """
        # ── Setup Figure ─────────────────────────────────────────
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(20, 24))
        fig.patch.set_facecolor("#0d1117")

        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

        colors = {
            "primary": "#00d4ff",
            "success": "#28a745",
            "danger": "#dc3545",
            "warning": "#ffc107",
            "purple": "#9b59b6",
            "background": "#161b22",
            "card": "#21262d",
        }

        # ── 1. Confusion Matrix Heatmap ───────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(self.y_test, self.y_pred)
        cm_labels = ["Legitimate", "Phishing"]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=sns.diverging_palette(10, 133, as_cmap=True),
            ax=ax1,
            xticklabels=cm_labels,
            yticklabels=cm_labels,
            linewidths=2,
            annot_kws={"size": 16, "weight": "bold"},
        )
        ax1.set_title("Confusion Matrix", color=colors["primary"],
                      fontsize=14, fontweight="bold", pad=15)
        ax1.set_xlabel("Predicted Label", color="white", fontsize=11)
        ax1.set_ylabel("True Label", color="white", fontsize=11)
        ax1.tick_params(colors="white")
        ax1.set_facecolor(colors["background"])

        # ── 2. ROC Curve ──────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        if self.y_prob is not None:
            fpr, tpr, thresholds = roc_curve(self.y_test, self.y_prob)
            roc_auc = auc(fpr, tpr)

            ax2.plot(
                fpr, tpr,
                color=colors["primary"],
                linewidth=3,
                label=f"ROC Curve (AUC = {roc_auc:.4f})"
            )
            ax2.fill_between(fpr, tpr, alpha=0.15, color=colors["primary"])
            ax2.plot(
                [0, 1], [0, 1],
                color=colors["danger"],
                linewidth=2,
                linestyle="--",
                label="Random Classifier",
            )
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel("False Positive Rate", color="white", fontsize=11)
            ax2.set_ylabel("True Positive Rate", color="white", fontsize=11)
            ax2.set_title("ROC Curve", color=colors["primary"],
                          fontsize=14, fontweight="bold", pad=15)
            ax2.legend(loc="lower right", fontsize=10,
                      facecolor=colors["card"], labelcolor="white")
            ax2.tick_params(colors="white")
            ax2.set_facecolor(colors["background"])
            ax2.grid(True, alpha=0.2)

        # ── 3. Precision-Recall Curve ─────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        if self.y_prob is not None:
            precision_vals, recall_vals, _ = precision_recall_curve(
                self.y_test, self.y_prob
            )
            pr_auc = auc(recall_vals, precision_vals)

            ax3.plot(
                recall_vals, precision_vals,
                color=colors["warning"],
                linewidth=3,
                label=f"PR Curve (AUC = {pr_auc:.4f})",
            )
            ax3.fill_between(recall_vals, precision_vals, alpha=0.15,
                            color=colors["warning"])
            ax3.set_xlabel("Recall", color="white", fontsize=11)
            ax3.set_ylabel("Precision", color="white", fontsize=11)
            ax3.set_title("Precision-Recall Curve", color=colors["primary"],
                          fontsize=14, fontweight="bold", pad=15)
            ax3.legend(fontsize=10, facecolor=colors["card"], labelcolor="white")
            ax3.tick_params(colors="white")
            ax3.set_facecolor(colors["background"])
            ax3.grid(True, alpha=0.2)

        # ── 4. Performance Metrics Bar Chart ──────────────────────
        ax4 = fig.add_subplot(gs[1, 0])
        metrics = self.calculate_metrics()
        metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
        metric_values = [
            metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1_score"]
        ]

        bar_colors = [
            colors["success"] if v >= 0.90 else
            colors["warning"] if v >= 0.80 else
            colors["danger"]
            for v in metric_values
        ]

        bars = ax4.bar(metric_names, metric_values, color=bar_colors,
                       edgecolor="white", linewidth=0.5, width=0.6)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{value:.3f}",
                ha="center", va="bottom",
                color="white", fontweight="bold", fontsize=12,
            )

        ax4.set_ylim([0, 1.12])
        ax4.set_title("Performance Metrics", color=colors["primary"],
                      fontsize=14, fontweight="bold", pad=15)
        ax4.set_ylabel("Score", color="white", fontsize=11)
        ax4.tick_params(colors="white")
        ax4.set_facecolor(colors["background"])
        ax4.axhline(y=0.9, color=colors["success"], linestyle="--",
                    alpha=0.5, linewidth=1.5, label="90% threshold")
        ax4.legend(fontsize=9, facecolor=colors["card"], labelcolor="white")
        ax4.grid(True, axis="y", alpha=0.2)

        # ── 5. Model Comparison ───────────────────────────────────
        ax5 = fig.add_subplot(gs[1, 1:])
        if model_results:
            model_names = list(model_results.keys())
            test_accs = [model_results[m]["test_accuracy"] for m in model_names]
            cv_means = [model_results[m]["cv_mean"] for m in model_names]
            cv_stds = [model_results[m]["cv_std"] for m in model_results]

            x = np.arange(len(model_names))
            width = 0.35

            bars1 = ax5.bar(
                x - width/2, test_accs, width,
                label="Test Accuracy",
                color=colors["primary"],
                alpha=0.85, edgecolor="white", linewidth=0.5,
            )
            bars2 = ax5.bar(
                x + width/2, cv_means, width,
                label="CV Mean Accuracy",
                color=colors["purple"],
                alpha=0.85, edgecolor="white", linewidth=0.5,
                yerr=cv_stds, capsize=5, error_kw={"color": "white"},
            )

            ax5.set_xticks(x)
            ax5.set_xticklabels(
                [m.replace(" ", "\n") for m in model_names],
                color="white", fontsize=9,
            )
            ax5.set_ylim([0.5, 1.05])
            ax5.set_title("Model Comparison", color=colors["primary"],
                          fontsize=14, fontweight="bold", pad=15)
            ax5.set_ylabel("Accuracy Score", color="white", fontsize=11)
            ax5.tick_params(colors="white")
            ax5.set_facecolor(colors["background"])
            ax5.legend(fontsize=10, facecolor=colors["card"], labelcolor="white")
            ax5.axhline(y=0.9, color=colors["success"], linestyle="--",
                       alpha=0.5, linewidth=1.5)
            ax5.grid(True, axis="y", alpha=0.2)

            # Add accuracy labels
            for bar in bars1:
                height = bar.get_height()
                ax5.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.003,
                    f"{height:.3f}",
                    ha="center", va="bottom",
                    color="white", fontsize=8, fontweight="bold",
                )

        # ── 6. Feature Importance ─────────────────────────────────
        ax6 = fig.add_subplot(gs[2, :])
        if feature_importance_df is not None:
            top_20 = feature_importance_df.head(20)

            colors_feat = [
                colors["danger"] if "phishing" in f or "urgent" in f or "threat" in f
                else colors["warning"] if "suspicious" in f or "money" in f
                else colors["primary"]
                for f in top_20["Feature"]
            ]

            bars = ax6.barh(
                range(len(top_20)),
                top_20["Importance"],
                color=colors_feat,
                edgecolor="none",
                height=0.7,
            )

            ax6.set_yticks(range(len(top_20)))
            ax6.set_yticklabels(top_20["Feature"], color="white", fontsize=10)
            ax6.set_xlabel("Importance Score", color="white", fontsize=11)
            ax6.set_title(
                "Top 20 Most Important Features",
                color=colors["primary"], fontsize=14, fontweight="bold", pad=15
            )
            ax6.tick_params(colors="white")
            ax6.set_facecolor(colors["background"])
            ax6.invert_yaxis()
            ax6.grid(True, axis="x", alpha=0.2)

            # Value labels
            for bar, value in zip(bars, top_20["Importance"]):
                ax6.text(
                    value + 0.001,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.4f}",
                    va="center", color="white", fontsize=8,
                )

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors["danger"], label="Phishing Indicators"),
                Patch(facecolor=colors["warning"], label="Suspicious Patterns"),
                Patch(facecolor=colors["primary"], label="General Features"),
            ]
            ax6.legend(handles=legend_elements, loc="lower right",
                      facecolor=colors["card"], labelcolor="white", fontsize=10)

        # ── 7. Prediction Distribution ────────────────────────────
        ax7 = fig.add_subplot(gs[3, 0])
        if self.y_prob is not None:
            phishing_probs = self.y_prob[self.y_test == 1]
            legit_probs = self.y_prob[self.y_test == 0]

            ax7.hist(
                legit_probs, bins=30, alpha=0.7,
                color=colors["success"], label="Legitimate",
                edgecolor="white", linewidth=0.3,
            )
            ax7.hist(
                phishing_probs, bins=30, alpha=0.7,
                color=colors["danger"], label="Phishing",
                edgecolor="white", linewidth=0.3,
            )
            ax7.axvline(x=0.5, color="white", linestyle="--",
                       linewidth=2, label="Decision Boundary (0.5)")
            ax7.set_xlabel("Phishing Probability", color="white", fontsize=11)
            ax7.set_ylabel("Count", color="white", fontsize=11)
            ax7.set_title("Prediction Probability Distribution",
                          color=colors["primary"], fontsize=12, fontweight="bold", pad=15)
            ax7.legend(fontsize=9, facecolor=colors["card"], labelcolor="white")
            ax7.tick_params(colors="white")
            ax7.set_facecolor(colors["background"])
            ax7.grid(True, alpha=0.2)

        # ── 8. Prediction Results Pie ─────────────────────────────
        ax8 = fig.add_subplot(gs[3, 1])
        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()

        pie_labels = ["True Negative\n(Legit ✓)", "False Positive\n(Legit as Phish ✗)",
                     "False Negative\n(Phish Missed ✗)", "True Positive\n(Phish ✓)"]
        pie_sizes = [tn, fp, fn, tp]
        pie_colors = [colors["success"], colors["warning"],
                     colors["danger"], colors["primary"]]
        explode = (0, 0.1, 0.1, 0)

        wedges, texts, autotexts = ax8.pie(
            pie_sizes,
            labels=pie_labels,
            colors=pie_colors,
            autopct="%1.1f%%",
            explode=explode,
            startangle=90,
            textprops={"color": "white", "fontsize": 9},
        )
        for autotext in autotexts:
            autotext.set_fontweight("bold")

        ax8.set_title("Prediction Breakdown", color=colors["primary"],
                     fontsize=12, fontweight="bold", pad=15)

        # ── 9. Summary Score Card ─────────────────────────────────
        ax9 = fig.add_subplot(gs[3, 2])
        ax9.set_facecolor(colors["background"])
        ax9.axis("off")

        metrics = self.calculate_metrics()
        overall_score = np.mean([
            metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1_score"]
        ])

        score_color = (colors["success"] if overall_score >= 0.90 else
                      colors["warning"] if overall_score >= 0.80 else
                      colors["danger"])

        summary_text = [
            ("OVERALL SCORE", f"{overall_score*100:.1f}%", 0.85, 24, score_color),
            ("Accuracy",  f"{metrics['accuracy']*100:.2f}%",  0.70, 14, "white"),
            ("Precision", f"{metrics['precision']*100:.2f}%", 0.60, 14, "white"),
            ("Recall",    f"{metrics['recall']*100:.2f}%",    0.50, 14, "white"),
            ("F1-Score",  f"{metrics['f1_score']*100:.2f}%",  0.40, 14, "white"),
        ]

        if metrics["auc_roc"]:
            summary_text.append(
                ("AUC-ROC", f"{metrics['auc_roc']*100:.2f}%", 0.30, 14, "white")
            )

        for label, value, y_pos, fontsize, color in summary_text:
            ax9.text(0.3, y_pos, label, transform=ax9.transAxes,
                    fontsize=fontsize, color=color, fontweight="bold",
                    ha="right", va="center")
            ax9.text(0.35, y_pos, value, transform=ax9.transAxes,
                    fontsize=fontsize, color=score_color if label == "OVERALL SCORE" else colors["primary"],
                    fontweight="bold", ha="left", va="center")

        grade = ("A+" if overall_score >= 0.95 else "A" if overall_score >= 0.90
                else "B" if overall_score >= 0.85 else "C")
        ax9.text(0.5, 0.15, f"Grade: {grade}", transform=ax9.transAxes,
                fontsize=20, color=score_color, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["card"],
                         edgecolor=score_color, linewidth=2))

        # ── Title ─────────────────────────────────────────────────
        fig.suptitle(
            "🎣 Phishing Email Detection - Complete Analysis Dashboard",
            fontsize=20, fontweight="bold",
            color=colors["primary"], y=0.98,
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        print(f"\n{Fore.GREEN}[✓] Visualization saved: {save_path}{Style.RESET_ALL}")
        plt.show()