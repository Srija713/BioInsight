"""
BioInsight Lite - Model Explainability
Uses SHAP for model interpretation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelExplainer:
    def __init__(self, model, X_train, X_test, feature_names, model_type='tree'):
        """
        Initialize explainer
        
        Args:
            model: Trained model
            X_train: Training data
            X_test: Test data
            feature_names: List of feature names
            model_type: 'tree' for tree-based, 'linear' for linear models
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self, background_samples=100):
        """Create SHAP explainer"""
        print(f"Creating SHAP explainer for {self.model_type} model...")
        
        if self.model_type == 'tree':
            # For tree-based models (RandomForest, XGBoost)
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # For linear models
            background = shap.sample(self.X_train, background_samples)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background
            )
        
        print("Explainer created successfully!")
    
    def calculate_shap_values(self, n_samples=None):
        """Calculate SHAP values"""
        if self.explainer is None:
            self.create_explainer()
        
        print("Calculating SHAP values...")
        
        # Use subset for faster computation
        if n_samples and n_samples < len(self.X_test):
            X_explain = self.X_test[:n_samples]
        else:
            X_explain = self.X_test
        
        if self.model_type == 'tree':
            self.shap_values = self.explainer.shap_values(X_explain)
            # For binary classification, get values for positive class
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]
        else:
            self.shap_values = self.explainer.shap_values(X_explain)
        
        self.X_explain = X_explain
        print(f"SHAP values calculated for {len(X_explain)} samples")
        
        return self.shap_values
    
    def plot_summary(self, save_path=None):
        """Create SHAP summary plot"""
        if self.shap_values is None:
            self.calculate_shap_values()
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_explain,
            feature_names=self.feature_names,
            show=False
        )
        plt.title("SHAP Summary Plot - Feature Impact on Predictions")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_bar_importance(self, save_path=None):
        """Create SHAP bar plot for feature importance"""
        if self.shap_values is None:
            self.calculate_shap_values()
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_explain,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        plt.title("SHAP Feature Importance - Mean Absolute Impact")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_waterfall(self, sample_idx=0):
        """Create SHAP waterfall plot for individual prediction"""
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # Create explanation object
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.explainer.expected_value if self.model_type == 'tree' 
                       else self.explainer.expected_value[1],
            data=self.X_explain[sample_idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.title(f"SHAP Waterfall Plot - Sample {sample_idx}")
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_force(self, sample_idx=0):
        """Create SHAP force plot for individual prediction"""
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # Force plot
        shap.initjs()
        
        base_value = (self.explainer.expected_value if self.model_type == 'tree' 
                     else self.explainer.expected_value[1])
        
        force_plot = shap.force_plot(
            base_value,
            self.shap_values[sample_idx],
            self.X_explain[sample_idx],
            feature_names=self.feature_names
        )
        
        return force_plot
    
    def get_feature_contributions(self, top_n=10):
        """Get top contributing features across all samples"""
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        return contributions.head(top_n)
    
    def plot_dependence(self, feature_idx=0, interaction_idx='auto'):
        """Create SHAP dependence plot"""
        if self.shap_values is None:
            self.calculate_shap_values()
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            self.X_explain,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            show=False
        )
        plt.title(f"SHAP Dependence Plot - {self.feature_names[feature_idx]}")
        plt.tight_layout()
        
        return plt.gcf()
    
    def explain_prediction(self, sample_idx=0):
        """Comprehensive explanation for a single prediction"""
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # Get feature values and SHAP values
        feature_values = self.X_explain[sample_idx]
        shap_vals = self.shap_values[sample_idx]
        
        # Create explanation DataFrame
        explanation_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Value': feature_values,
            'SHAP Value': shap_vals,
            'Abs SHAP': np.abs(shap_vals)
        }).sort_values('Abs SHAP', ascending=False)
        
        base_value = (self.explainer.expected_value if self.model_type == 'tree' 
                     else self.explainer.expected_value[1])
        
        prediction = base_value + shap_vals.sum()
        
        print(f"\nPrediction Explanation for Sample {sample_idx}")
        print("=" * 70)
        print(f"Base value (average prediction): {base_value:.4f}")
        print(f"Final prediction: {prediction:.4f}")
        print(f"Prediction probability: {1 / (1 + np.exp(-prediction)):.4f}")
        print("\nTop Contributing Features:")
        print(explanation_df.head(10).to_string(index=False))
        
        return explanation_df
    
    def create_interactive_summary(self):
        """Create interactive Plotly visualization of SHAP values"""
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        feature_order = np.argsort(mean_abs_shap)[::-1][:15]  # Top 15
        
        # Create violin plots for each feature
        fig = go.Figure()
        
        for idx in feature_order:
            fig.add_trace(go.Violin(
                y=self.shap_values[:, idx],
                name=self.feature_names[idx],
                box_visible=True,
                meanline_visible=True
            ))
        
        fig.update_layout(
            title="SHAP Value Distribution by Feature",
            yaxis_title="SHAP Value",
            xaxis_title="Feature",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def generate_report(self, output_dir='explainability_report'):
        """Generate comprehensive explainability report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating Explainability Report...")
        
        # Calculate SHAP values if not done
        if self.shap_values is None:
            self.calculate_shap_values(n_samples=500)
        
        # 1. Summary plot
        summary_fig = self.plot_summary()
        summary_fig.savefig(os.path.join(output_dir, 'shap_summary.png'), 
                           dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bar importance
        bar_fig = self.plot_bar_importance()
        bar_fig.savefig(os.path.join(output_dir, 'shap_importance.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Top feature contributions
        contributions = self.get_feature_contributions(top_n=15)
        contributions.to_csv(os.path.join(output_dir, 'feature_contributions.csv'), 
                            index=False)
        
        # 4. Sample explanations
        for i in range(min(3, len(self.X_explain))):
            waterfall_fig = self.plot_waterfall(sample_idx=i)
            waterfall_fig.savefig(
                os.path.join(output_dir, f'waterfall_sample_{i}.png'),
                dpi=300, bbox_inches='tight'
            )
            plt.close()
            
            explanation_df = self.explain_prediction(sample_idx=i)
            explanation_df.to_csv(
                os.path.join(output_dir, f'explanation_sample_{i}.csv'),
                index=False
            )
        
        # 5. Interactive summary
        interactive_fig = self.create_interactive_summary()
        interactive_fig.write_html(os.path.join(output_dir, 'interactive_summary.html'))
        
        print(f"\nExplainability report saved to {output_dir}/")
        print("Generated files:")
        print("  - shap_summary.png")
        print("  - shap_importance.png")
        print("  - feature_contributions.csv")
        print("  - waterfall_sample_*.png (3 samples)")
        print("  - explanation_sample_*.csv (3 samples)")
        print("  - interactive_summary.html")


# Example usage (safe guard)
if __name__ == "__main__":
    import joblib
    import os

    try:
        model_path = 'models/xgboost.pkl'
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}. Skipping example usage.")
        else:
            model = joblib.load(model_path)

            # Try to import dataset variables from `ml_models` if available
            try:
                from ml_models import X_train, X_test, feature_names  # optional
            except Exception:
                X_train = globals().get('X_train', None)
                X_test = globals().get('X_test', None)
                feature_names = globals().get('feature_names', None)

            if X_train is None or X_test is None or feature_names is None:
                print("X_train, X_test, or feature_names not available. Skipping example usage.")
            else:
                explainer = ModelExplainer(
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    feature_names=feature_names,
                    model_type='tree'
                )

                # Generate report and show a couple of plots
                explainer.generate_report()
                summary_fig = explainer.plot_summary()
                plt.show()

                waterfall_fig = explainer.plot_waterfall(sample_idx=0)
                plt.show()

                contributions = explainer.get_feature_contributions()
                print(contributions)
    except Exception as e:
        print(f"Skipped example usage due to error: {e}")