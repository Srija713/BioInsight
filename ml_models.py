"""
BioInsight Lite - ML Models for Bioactivity Prediction
Implements multiple models with comprehensive evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime

class BioactivityPredictor:
    def __init__(self, X, y, feature_names, test_size=0.2, random_state=42):
        """
        Initialize predictor with data
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            test_size: Proportion of test set
            random_state: Random seed
        """
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.test_size = test_size
        self.random_state = random_state
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Store models and results
        self.models = {}
        self.results = {}
        
        print(f"Data split:")
        print(f"  Training: {self.X_train.shape[0]} samples")
        print(f"  Testing: {self.X_test.shape[0]} samples")
        print(f"  Features: {self.X_train.shape[1]}")
    
    def train_logistic_regression(self):
        """Train Logistic Regression baseline"""
        print("\n" + "="*60)
        print("Training Logistic Regression")
        print("="*60)
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        model.fit(self.X_train_scaled, self.y_train)
        
        # Predictions
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Evaluate
        results = self._evaluate_model(y_pred, y_pred_proba, "Logistic Regression")
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        results['feature_importance'] = feature_importance
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        
        return model, results
    
    def train_random_forest(self, n_estimators=100):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("Training Random Forest")
        print("="*60)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        results = self._evaluate_model(y_pred, y_pred_proba, "Random Forest")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results['feature_importance'] = feature_importance
        
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        return model, results

    def tune_random_forest(self, param_distributions=None, n_iter=30, cv=5, scoring='f1', n_jobs=-1, oversample=False):
        """Hyperparameter tuning for Random Forest using RandomizedSearchCV

        Args:
            param_distributions: dict of parameters to sample from
            n_iter: number of parameter settings that are sampled
            cv: number of CV folds
            scoring: scoring metric for CV
            n_jobs: parallel jobs
            oversample: whether to oversample minority class in training data before tuning
        """
        print("\n" + "="*60)
        print("Tuning Random Forest (RandomizedSearchCV)")
        print("="*60)

        if oversample:
            self._oversample_training()

        if param_distributions is None:
            param_distributions = {
                'n_estimators': [100, 300, 500],
                'max_depth': [None, 6, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }

        base = RandomForestClassifier(random_state=self.random_state, class_weight='balanced', n_jobs=-1)

        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv_strategy,
            random_state=self.random_state,
            n_jobs=n_jobs,
            verbose=1
        )

        search.fit(self.X_train, self.y_train)

        best = search.best_estimator_
        print(f"Best RF params: {search.best_params_}")

        # Evaluate on test set
        y_pred = best.predict(self.X_test)
        y_pred_proba = best.predict_proba(self.X_test)[:, 1]
        results = self._evaluate_model(y_pred, y_pred_proba, "Random Forest (Tuned)")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': best.feature_importances_
        }).sort_values('importance', ascending=False)

        results['feature_importance'] = feature_importance

        self.models['random_forest_tuned'] = best
        self.results['random_forest_tuned'] = results
        self.results['random_forest_tuned']['cv_results'] = search.cv_results_

        return best, results
    
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("Training XGBoost")
        print("="*60)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        results = self._evaluate_model(y_pred, y_pred_proba, "XGBoost")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results['feature_importance'] = feature_importance
        
        self.models['xgboost'] = model
        self.results['xgboost'] = results
        
        return model, results

    def tune_xgboost(self, param_distributions=None, n_iter=30, cv=5, scoring='f1', n_jobs=-1, oversample=False):
        """Hyperparameter tuning for XGBoost using RandomizedSearchCV"""
        print("\n" + "="*60)
        print("Tuning XGBoost (RandomizedSearchCV)")
        print("="*60)

        if oversample:
            self._oversample_training()

        # default param space
        if param_distributions is None:
            param_distributions = {
                'n_estimators': [100, 200, 400],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 1, 5]
            }

        base = xgb.XGBClassifier(
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv_strategy,
            random_state=self.random_state,
            n_jobs=n_jobs,
            verbose=1
        )

        search.fit(self.X_train, self.y_train)

        best = search.best_estimator_
        print(f"Best XGB params: {search.best_params_}")

        # Evaluate on test set
        y_pred = best.predict(self.X_test)
        y_pred_proba = best.predict_proba(self.X_test)[:, 1]
        results = self._evaluate_model(y_pred, y_pred_proba, "XGBoost (Tuned)")

        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': best.feature_importances_
        }).sort_values('importance', ascending=False)

        results['feature_importance'] = feature_importance

        self.models['xgboost_tuned'] = best
        self.results['xgboost_tuned'] = results
        self.results['xgboost_tuned']['cv_results'] = search.cv_results_

        return best, results

    def cross_validate_best(self, model_name, cv=5, scoring='f1'):
        """Run cross-validation on the specified trained model using training data"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found for CV")

        model = self.models[model_name]
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv_strategy, scoring=scoring, n_jobs=-1)
        print(f"Cross-validation {scoring} scores ({model_name}): {scores}")
        print(f"Mean {scoring}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores

    def _oversample_training(self, random_state=None):
        """Simple random oversampling of minority class in training set."""
        print("Performing simple random oversampling on training data...")
        try:
            X_train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        except Exception:
            X_train_df = pd.DataFrame(self.X_train)

        y_train_ser = pd.Series(self.y_train, name='is_active')
        train_df = pd.concat([X_train_df, y_train_ser.reset_index(drop=True)], axis=1)

        counts = train_df['is_active'].value_counts()
        if counts.min() == counts.max():
            print('Classes already balanced. Skipping oversampling.')
            return

        majority_class = counts.idxmax()
        majority_count = counts.max()

        dfs = [train_df[train_df['is_active'] == cls] for cls in counts.index]
        resampled = []
        for cls_df in dfs:
            if len(cls_df) < majority_count:
                resampled_cls = cls_df.sample(n=majority_count, replace=True, random_state=random_state or self.random_state)
            else:
                resampled_cls = cls_df
            resampled.append(resampled_cls)

        balanced_df = pd.concat(resampled).sample(frac=1, random_state=random_state or self.random_state).reset_index(drop=True)

        y_bal = balanced_df['is_active'].values
        X_bal = balanced_df.drop(columns=['is_active']).values

        # Update training sets and scaled versions
        self.X_train = X_bal
        self.y_train = y_bal
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)

        print('Oversampling complete. New training size:', self.X_train.shape)
    
    def _evaluate_model(self, y_pred, y_pred_proba, model_name):
        """Comprehensive model evaluation"""
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"\n{model_name} Performance:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")
        print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(results['confusion_matrix'])
        
        return results
    
    def compare_models(self):
        """Compare all trained models"""
        if not self.results:
            print("No models trained yet!")
            return None
        
        comparison = pd.DataFrame({
            'Model': [r['model_name'] for r in self.results.values()],
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'Precision': [r['precision'] for r in self.results.values()],
            'Recall': [r['recall'] for r in self.results.values()],
            'F1-Score': [r['f1_score'] for r in self.results.values()],
            'ROC-AUC': [r['roc_auc'] for r in self.results.values()]
        })
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(comparison.to_string(index=False))
        
        # Identify best models
        best_f1 = comparison.loc[comparison['F1-Score'].idxmax(), 'Model']
        best_auc = comparison.loc[comparison['ROC-AUC'].idxmax(), 'Model']
        
        print(f"\nBest F1-Score: {best_f1}")
        print(f"Best ROC-AUC: {best_auc}")
        
        return comparison
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        fig = go.Figure()
        
        for model_name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            auc_score = results['roc_auc']
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{results['model_name']} (AUC={auc_score:.3f})",
                mode='lines',
                line=dict(width=2)
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random (AUC=0.500)',
            mode='lines',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='ROC Curves - Model Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800,
            height=600,
            hovermode='closest'
        )
        
        return fig
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Inactive', 'Active'],
                yticklabels=['Inactive', 'Active'],
                ax=axes[idx]
            )
            axes[idx].set_title(f"{results['model_name']}\nConfusion Matrix")
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, model_name, top_n=15):
        """Plot feature importance for a specific model"""
        if model_name not in self.results:
            print(f"Model {model_name} not found!")
            return None
        
        fi = self.results[model_name]['feature_importance'].head(top_n)
        
        fig = go.Figure(go.Bar(
            x=fi['importance'] if 'importance' in fi.columns else abs(fi['coefficient']),
            y=fi['feature'],
            orientation='h',
            marker=dict(color='steelblue')
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Features - {self.results[model_name]["model_name"]}',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def save_models(self, output_dir='models'):
        """Save trained models and results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f"{model_name}.pkl")
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(output_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature names
        with open(os.path.join(output_dir, "feature_names.json"), 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save results (without predictions)
        results_summary = {}
        for model_name, results in self.results.items():
            results_summary[model_name] = {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'roc_auc': float(results['roc_auc'])
            }
        
        with open(os.path.join(output_dir, "results_summary.json"), 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\nAll models and results saved to {output_dir}/")
    
    def predict(self, X_new, model_name='xgboost'):
        """Make predictions on new data"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found!")
        
        model = self.models[model_name]
        
        # Scale if logistic regression
        if model_name == 'logistic_regression':
            X_new_processed = self.scaler.transform(X_new)
        else:
            X_new_processed = X_new
        
        predictions = model.predict(X_new_processed)
        probabilities = model.predict_proba(X_new_processed)
        
        return predictions, probabilities


# Example usage
if __name__ == "__main__":
    # Assuming X, y, feature_names from preprocessing
    from eda_pipeline import BioactivityPreprocessor
    
    df = pd.read_csv('bioactivity_data.csv')
    preprocessor = BioactivityPreprocessor(df)
    preprocessor.create_target_variable()
    preprocessor.clean_and_engineer_features()
    X, y, feature_names = preprocessor.prepare_ml_features()
    
    # Initialize predictor
    predictor = BioactivityPredictor(X, y, feature_names)
    
    # Train models
    lr_model, lr_results = predictor.train_logistic_regression()
    rf_model, rf_results = predictor.train_random_forest()
    xgb_model, xgb_results = predictor.train_xgboost()
    
    # Compare models
    comparison = predictor.compare_models()
    
    # Visualizations
    roc_fig = predictor.plot_roc_curves()
    roc_fig.show()
    
    cm_fig = predictor.plot_confusion_matrices()
    plt.show()
    
    # Feature importance
    fi_fig = predictor.plot_feature_importance('xgboost')
    fi_fig.show()
    
    # Save models
    predictor.save_models()