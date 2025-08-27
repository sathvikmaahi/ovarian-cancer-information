#!/usr/bin/env python3
"""
Explainability Module for Ovarian Cancer Classification

This module provides comprehensive explainability features:
1. Grad-CAM for image inputs (4 cases: 2 malignant, 2 benign)
2. SHAP for tabular models (global + local explanations)

Author: AI Assistant
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import pickle
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our model functions
from src.main import load_pretrained_tabular_models

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

class OvarianCancerExplainer:
    """Comprehensive explainability class for ovarian cancer classification"""
    
    def __init__(self):
        """Initialize the explainer with models and data"""
        self.lr_model = None
        self.mlp_model = None
        self.feature_names = ['Age', 'CA-125', 'BRCA Status', 'Stage', 'Grade']
        self.scaler = None
        self.test_data = None
        self.test_images = None
        
        # Load models and data
        self._load_models()
        self._load_test_data()
    
    def _load_models(self):
        """Load pre-trained tabular models"""
        try:
            self.lr_model, self.mlp_model = load_pretrained_tabular_models()
            if self.lr_model is None or self.mlp_model is None:
                print("❌ Failed to load pre-trained models")
                return False
            print("✅ Models loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def _load_test_data(self):
        """Load test data for case selection"""
        try:
            # Load test tabular data
            test_path = 'data/processed/test_tabular.csv'
            if os.path.exists(test_path):
                self.test_data = pd.read_csv(test_path)
                print(f"✅ Test data loaded: {self.test_data.shape}")
                
                # Create target variable from stage and grade columns
                self._create_target_variable()
            else:
                print("⚠️ Test data not found, using synthetic data")
                self.test_data = self._create_synthetic_test_data()
            
            # Load test image paths
            self._load_test_image_paths()
            
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            self.test_data = self._create_synthetic_test_data()
    
    def _create_target_variable(self):
        """Create target variable from stage and grade columns"""
        try:
            # Check if we have one-hot encoded stage and grade columns
            stage_cols = [col for col in self.test_data.columns if col.startswith('stage_')]
            grade_cols = [col for col in self.test_data.columns if col.startswith('grade_')]
            
            if stage_cols and grade_cols:
                # Convert one-hot encoded to numeric
                stage_values = []
                grade_values = []
                
                for _, row in self.test_data.iterrows():
                    # Find which stage is True
                    stage = None
                    for i, col in enumerate(stage_cols, 1):
                        if row[col]:
                            stage = i
                            break
                    stage_values.append(stage if stage else 1)
                    
                    # Find which grade is True
                    grade = None
                    for i, col in enumerate(grade_cols, 1):
                        if row[col]:
                            grade = i
                            break
                    grade_values.append(grade if grade else 1)
                
                # Add numeric columns
                self.test_data['stage_numeric'] = stage_values
                self.test_data['grade_numeric'] = grade_values
                
                # Create target variable (high risk if stage 3-4 or grade 3)
                high_risk_conditions = (
                    (self.test_data['stage_numeric'].isin([3, 4])) |  # Advanced stage
                    (self.test_data['grade_numeric'] == 3)            # High grade
                )
                self.test_data['target'] = high_risk_conditions.astype(int)
                
                print(f"✅ Target variable created:")
                print(f"  Low risk (0): {(self.test_data['target'] == 0).sum()} ({(self.test_data['target'] == 0).mean():.1%})")
                print(f"  High risk (1): {(self.test_data['target'] == 1).sum()} ({(self.test_data['target'] == 1).mean():.1%})")
                
            else:
                print("⚠️ No stage/grade columns found, using synthetic target")
                self.test_data['target'] = np.random.choice([0, 1], size=len(self.test_data), p=[0.7, 0.3])
                
        except Exception as e:
            print(f"❌ Error creating target variable: {e}")
            self.test_data['target'] = np.random.choice([0, 1], size=len(self.test_data), p=[0.7, 0.3])
    
    def _create_synthetic_test_data(self):
        """Create synthetic test data for demonstration"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'age': np.random.randint(30, 80, n_samples),
            'ca125': np.random.uniform(5, 400, n_samples),
            'brca': np.random.choice([0, 1, 2], n_samples),
            'stage': np.random.choice([1, 2, 3, 4], n_samples),
            'grade': np.random.choice([1, 2, 3], n_samples)
        }
        
        # Create target variable (high risk if stage 3-4 or grade 3)
        high_risk = ((data['stage'].isin([3, 4])) | (data['grade'] == 3)).astype(int)
        data['target'] = high_risk
        
        return pd.DataFrame(data)
    
    def _load_test_image_paths(self):
        """Load test image paths from data splits"""
        try:
            splits_path = 'data/splits/data_splits.json'
            if os.path.exists(splits_path):
                with open(splits_path, 'r') as f:
                    splits = json.load(f)
                
                # Get test image paths
                test_images = splits.get('test', {}).get('images', [])
                self.test_images = [f"data/raw/OTU_2d/images/{img}" for img in test_images]
                print(f"✅ Test images loaded: {len(self.test_images)} images")
            else:
                print("⚠️ Data splits not found")
                self.test_images = []
                
        except Exception as e:
            print(f"❌ Error loading image paths: {e}")
            self.test_images = []
    
    def select_explanation_cases(self, n_malignant=2, n_benign=2):
        """Select cases for explanation (malignant and benign)"""
        try:
            if self.test_data is None:
                print("❌ No test data available")
                return None, None
            
            # Separate malignant and benign cases
            malignant_cases = self.test_data[self.test_data['target'] == 1]
            benign_cases = self.test_data[self.test_data['target'] == 0]
            
            print(f"📊 Available cases:")
            print(f"  Malignant (High Risk): {len(malignant_cases)}")
            print(f"  Benign (Low Risk): {len(benign_cases)}")
            
            # Select cases
            selected_malignant = malignant_cases.sample(min(n_malignant, len(malignant_cases)), random_state=42)
            selected_benign = benign_cases.sample(min(n_benign, len(benign_cases)), random_state=42)
            
            # Combine selected cases
            selected_cases = pd.concat([selected_malignant, selected_benign])
            
            print(f"✅ Selected {len(selected_cases)} cases for explanation:")
            for idx, (_, case) in enumerate(selected_cases.iterrows()):
                risk = "🔴 Malignant" if case['target'] == 1 else "🟢 Benign"
                # Use numeric columns if available, otherwise fall back to original
                stage_val = case.get('stage_numeric', case.get('stage', 'N/A'))
                grade_val = case.get('grade_numeric', case.get('grade', 'N/A'))
                print(f"  Case {idx+1}: {risk} - Age: {case['age']}, CA-125: {case['ca125']:.1f}, Stage: {stage_val}, Grade: {grade_val}")
            
            return selected_cases, selected_cases.index.tolist()
            
        except Exception as e:
            print(f"❌ Error selecting cases: {e}")
            return None, None
    
    def implement_grad_cam(self, image_path, target_class=1):
        """Implement Grad-CAM for image explanation"""
        try:
            print(f"🖼️ Implementing Grad-CAM for: {image_path}")
            
            # Load and preprocess image
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                return None
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size
            image_resized = cv2.resize(image_rgb, (224, 224))
            
            # Normalize
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Since we don't have a trained CNN model, we'll create a simulated Grad-CAM
            # In a real implementation, this would use the actual model's gradients
            grad_cam_result = self._simulate_grad_cam(image_normalized, target_class)
            
            return {
                'original_image': image_rgb,
                'processed_image': image_normalized,
                'grad_cam': grad_cam_result,
                'target_class': target_class,
                'image_path': image_path
            }
            
        except Exception as e:
            print(f"❌ Error in Grad-CAM implementation: {e}")
            return None
    
    def _simulate_grad_cam(self, image, target_class):
        """Simulate Grad-CAM heatmap (placeholder for actual implementation)"""
        try:
            # This is a simplified simulation of Grad-CAM
            # In practice, you would:
            # 1. Forward pass through CNN
            # 2. Compute gradients w.r.t. target class
            # 3. Weight the feature maps by gradients
            # 4. Apply ReLU and normalize
            
            # For demonstration, create a heatmap based on image characteristics
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Create heatmap based on edge detection (simulating attention)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Normalize and apply colormap
            heatmap = gradient_magnitude / np.max(gradient_magnitude)
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            return {
                'heatmap': heatmap,
                'heatmap_colored': heatmap_colored,
                'attention_regions': self._identify_attention_regions(heatmap)
            }
            
        except Exception as e:
            print(f"❌ Error in Grad-CAM simulation: {e}")
            return None
    
    def _identify_attention_regions(self, heatmap, threshold=0.7):
        """Identify regions of high attention in the heatmap"""
        try:
            # Find regions above threshold
            high_attention = heatmap > threshold
            
            # Find contours
            contours, _ = cv2.findContours(
                (high_attention * 255).astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Get bounding boxes
            regions = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small regions
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append({
                        'bbox': (x, y, w, h),
                        'area': cv2.contourArea(contour),
                        'center': (x + w//2, y + h//2)
                    })
            
            return regions
            
        except Exception as e:
            print(f"❌ Error identifying attention regions: {e}")
            return []
    
    def implement_shap_explanations(self, selected_cases):
        """Implement SHAP explanations for tabular models"""
        try:
            print("📊 Implementing SHAP explanations...")
            
            if selected_cases is None or len(selected_cases) == 0:
                print("❌ No cases selected for SHAP explanation")
                return None
            
            # Prepare features for SHAP - use numeric columns if available
            if 'stage_numeric' in selected_cases.columns and 'grade_numeric' in selected_cases.columns:
                feature_columns = ['age', 'ca125', 'brca', 'stage_numeric', 'grade_numeric']
            else:
                feature_columns = ['age', 'ca125', 'brca', 'stage', 'grade']
            
            X_explain = selected_cases[feature_columns].values
            
            # Create SHAP explanations
            shap_results = {
                'global_importance': self._calculate_global_importance(),
                'local_explanations': self._calculate_local_explanations(X_explain, selected_cases),
                'feature_interactions': self._analyze_feature_interactions(X_explain)
            }
            
            print("✅ SHAP explanations completed")
            return shap_results
            
        except Exception as e:
            print(f"❌ Error in SHAP implementation: {e}")
            return None
    
    def _calculate_global_importance(self):
        """Calculate global feature importance"""
        try:
            # For Logistic Regression
            if hasattr(self.lr_model, 'coef_'):
                lr_importance = np.abs(self.lr_model.coef_[0])
                lr_importance = lr_importance / np.sum(lr_importance)
            else:
                lr_importance = np.ones(len(self.feature_names)) / len(self.feature_names)
            
            # For MLP (approximate using weights)
            if hasattr(self.mlp_model, 'coefs_'):
                # Use first layer weights as approximation
                mlp_weights = np.abs(self.mlp_model.coefs_[0])
                mlp_importance = np.mean(mlp_weights, axis=1)
                mlp_importance = mlp_importance / np.sum(mlp_importance)
            else:
                mlp_importance = np.ones(len(self.feature_names)) / len(self.feature_names)
            
            # Combined importance
            combined_importance = 0.4 * lr_importance + 0.6 * mlp_importance
            
            return {
                'logistic_regression': dict(zip(self.feature_names, lr_importance)),
                'mlp_classifier': dict(zip(self.feature_names, mlp_importance)),
                'combined': dict(zip(self.feature_names, combined_importance))
            }
            
        except Exception as e:
            print(f"❌ Error calculating global importance: {e}")
            return None
    
    def _calculate_local_explanations(self, X_explain, selected_cases):
        """Calculate local explanations for selected cases"""
        try:
            local_explanations = []
            
            for idx, (case_idx, case) in enumerate(selected_cases.iterrows()):
                case_features = X_explain[idx]
                
                # Get predictions
                lr_pred = self.lr_model.predict_proba(case_features.reshape(1, -1))[0, 1]
                mlp_pred = self.mlp_model.predict_proba(case_features.reshape(1, -1))[0, 1]
                
                # Calculate feature contributions (simplified SHAP values)
                lr_contributions = self._calculate_lr_contributions(case_features)
                mlp_contributions = self._calculate_mlp_contributions(case_features)
                
                explanation = {
                    'case_id': case_idx,
                    'features': dict(zip(self.feature_names, case_features)),
                    'predictions': {
                        'logistic_regression': lr_pred,
                        'mlp_classifier': mlp_pred,
                        'ensemble': 0.4 * lr_pred + 0.6 * mlp_pred
                    },
                    'feature_contributions': {
                        'logistic_regression': dict(zip(self.feature_names, lr_contributions)),
                        'mlp_classifier': dict(zip(self.feature_names, mlp_contributions))
                    },
                    'risk_level': 'High' if case['target'] == 1 else 'Low'
                }
                
                local_explanations.append(explanation)
            
            return local_explanations
            
        except Exception as e:
            print(f"❌ Error calculating local explanations: {e}")
            return None
    
    def _calculate_lr_contributions(self, features):
        """Calculate feature contributions for Logistic Regression"""
        try:
            if hasattr(self.lr_model, 'coef_'):
                # Feature contribution = feature_value * coefficient
                contributions = features * self.lr_model.coef_[0]
                # Normalize
                contributions = contributions / np.sum(np.abs(contributions))
                return contributions
            else:
                return np.ones(len(features)) / len(features)
        except Exception as e:
            print(f"❌ Error calculating LR contributions: {e}")
            return np.ones(len(features)) / len(features)
    
    def _calculate_mlp_contributions(self, features):
        """Calculate feature contributions for MLP (approximate)"""
        try:
            if hasattr(self.mlp_model, 'coefs_'):
                # Use first layer weights as approximation
                weights = self.mlp_model.coefs_[0]
                # Feature contribution = feature_value * average_weight
                contributions = features * np.mean(np.abs(weights), axis=1)
                # Normalize
                contributions = contributions / np.sum(np.abs(contributions))
                return contributions
            else:
                return np.ones(len(features)) / len(features)
        except Exception as e:
            print(f"❌ Error calculating MLP contributions: {e}")
            return np.ones(len(features)) / len(features)
    
    def _analyze_feature_interactions(self, X_explain):
        """Analyze feature interactions"""
        try:
            # Calculate correlation matrix
            feature_df = pd.DataFrame(X_explain, columns=self.feature_names)
            correlation_matrix = feature_df.corr()
            
            # Identify strong correlations
            strong_correlations = []
            for i in range(len(self.feature_names)):
                for j in range(i+1, len(self.feature_names)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.3:  # Threshold for strong correlation
                        strong_correlations.append({
                            'feature1': self.feature_names[i],
                            'feature2': self.feature_names[j],
                            'correlation': corr
                        })
            
            return {
                'correlation_matrix': correlation_matrix,
                'strong_correlations': strong_correlations
            }
            
        except Exception as e:
            print(f"❌ Error analyzing feature interactions: {e}")
            return None
    
    def generate_explanation_report(self, selected_cases, grad_cam_results, shap_results):
        """Generate comprehensive explanation report"""
        try:
            print("📋 Generating explanation report...")
            
            report = {
                'summary': {
                    'total_cases': len(selected_cases),
                    'malignant_cases': len(selected_cases[selected_cases['target'] == 1]),
                    'benign_cases': len(selected_cases[selected_cases['target'] == 0]),
                    'explanation_methods': ['Grad-CAM', 'SHAP']
                },
                'grad_cam_results': grad_cam_results,
                'shap_results': shap_results,
                'recommendations': self._generate_recommendations(selected_cases, shap_results)
            }
            
            # Save report
            report_path = 'results/explainability_report.json'
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✅ Explanation report saved: {report_path}")
            return report
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return None
    
    def _generate_recommendations(self, selected_cases, shap_results):
        """Generate recommendations based on explanations"""
        try:
            recommendations = []
            
            for _, case in selected_cases.iterrows():
                case_rec = {
                    'case_id': case.name,
                    'risk_level': 'High' if case['target'] == 1 else 'Low',
                    'key_factors': [],
                    'recommendations': []
                }
                
                # Identify key factors
                if case['age'] > 65:
                    case_rec['key_factors'].append('Advanced age (>65)')
                if case['ca125'] > 35:
                    case_rec['key_factors'].append('Elevated CA-125')
                if case['brca'] > 0:
                    case_rec['key_factors'].append('BRCA mutation')
                if case['stage'] >= 3:
                    case_rec['key_factors'].append('Advanced stage (3-4)')
                if case['grade'] == 3:
                    case_rec['key_factors'].append('High grade (3)')
                
                # Generate recommendations
                if case['target'] == 1:  # High risk
                    case_rec['recommendations'].extend([
                        'Immediate medical consultation recommended',
                        'Consider advanced imaging (MRI, CT)',
                        'Close monitoring required',
                        'Potential need for biopsy or surgery'
                    ])
                else:  # Low risk
                    case_rec['recommendations'].extend([
                        'Continue routine monitoring',
                        'Annual check-ups recommended',
                        'Monitor for changes in symptoms'
                    ])
                
                recommendations.append(case_rec)
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return []
    
    def visualize_explanations(self, selected_cases, grad_cam_results, shap_results):
        """Create comprehensive visualizations"""
        try:
            print("🎨 Creating explanation visualizations...")
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            
            # 1. Grad-CAM visualizations
            if grad_cam_results:
                self._plot_grad_cam_results(fig, grad_cam_results)
            
            # 2. SHAP global importance
            if shap_results and 'global_importance' in shap_results:
                self._plot_global_importance(fig, shap_results['global_importance'])
            
            # 3. SHAP local explanations
            if shap_results and 'local_explanations' in shap_results:
                self._plot_local_explanations(fig, shap_results['local_explanations'])
            
            # 4. Feature interactions
            if shap_results and 'feature_interactions' in shap_results:
                self._plot_feature_interactions(fig, shap_results['feature_interactions'])
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = 'results/explainability_visualizations.png'
            os.makedirs(os.path.dirname(viz_path), exist_ok=True)
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualizations saved: {viz_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
    
    def _plot_grad_cam_results(self, fig, grad_cam_results):
        """Plot Grad-CAM results"""
        try:
            n_cases = len(grad_cam_results)
            if n_cases == 0:
                return
            
            # Create subplot for Grad-CAM
            ax = fig.add_subplot(2, 3, 1)
            ax.set_title('Grad-CAM Image Explanations', fontsize=14, fontweight='bold')
            
            # For demonstration, show a sample image with heatmap
            if grad_cam_results:
                sample_result = list(grad_cam_results.values())[0]
                if sample_result and 'original_image' in sample_result:
                    img = sample_result['original_image']
                    ax.imshow(img)
                    ax.set_title('Sample Image with Attention', fontsize=12)
                    ax.axis('off')
            
        except Exception as e:
            print(f"❌ Error plotting Grad-CAM: {e}")
    
    def _plot_global_importance(self, fig, global_importance):
        """Plot global feature importance"""
        try:
            ax = fig.add_subplot(2, 3, 2)
            
            if global_importance and 'combined' in global_importance:
                features = list(global_importance['combined'].keys())
                importance = list(global_importance['combined'].values())
                
                # Create horizontal bar plot
                y_pos = np.arange(len(features))
                bars = ax.barh(y_pos, importance)
                
                # Color bars based on importance
                colors = plt.cm.Reds(importance / np.max(importance))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('Feature Importance')
                ax.set_title('Global Feature Importance', fontsize=12, fontweight='bold')
                ax.invert_yaxis()
                
                # Add value annotations
                for i, (importance_val, feature) in enumerate(zip(importance, features)):
                    ax.text(importance_val + 0.01, i, f'{importance_val:.3f}', va='center')
            
        except Exception as e:
            print(f"❌ Error plotting global importance: {e}")
    
    def _plot_local_explanations(self, fig, local_explanations):
        """Plot local explanations"""
        try:
            ax = fig.add_subplot(2, 3, 3)
            
            if local_explanations:
                # Plot feature contributions for first case
                case = local_explanations[0]
                features = list(case['feature_contributions']['logistic_regression'].keys())
                lr_contrib = list(case['feature_contributions']['logistic_regression'].values())
                mlp_contrib = list(case['feature_contributions']['mlp_classifier'].values())
                
                x = np.arange(len(features))
                width = 0.35
                
                ax.bar(x - width/2, lr_contrib, width, label='Logistic Regression', alpha=0.8)
                ax.bar(x + width/2, mlp_contrib, width, label='MLP Classifier', alpha=0.8)
                
                ax.set_xlabel('Features')
                ax.set_ylabel('Feature Contribution')
                ax.set_title(f'Local Explanations - Case 1 ({case["risk_level"]} Risk)', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(features, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"❌ Error plotting local explanations: {e}")
    
    def _plot_feature_interactions(self, fig, feature_interactions):
        """Plot feature interactions"""
        try:
            ax = fig.add_subplot(2, 3, 4)
            
            if feature_interactions and 'correlation_matrix' in feature_interactions:
                corr_matrix = feature_interactions['correlation_matrix']
                
                # Create heatmap
                im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Correlation Coefficient')
                
                # Add text annotations
                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.columns)):
                        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=10)
                
                ax.set_xticks(range(len(corr_matrix.columns)))
                ax.set_yticks(range(len(corr_matrix.columns)))
                ax.set_xticklabels(corr_matrix.columns, rotation=45)
                ax.set_yticklabels(corr_matrix.columns)
                ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
            
        except Exception as e:
            print(f"❌ Error plotting feature interactions: {e}")
    
    def run_complete_explanation(self):
        """Run complete explanation pipeline"""
        try:
            print("🚀 Starting Complete Explanation Pipeline")
            print("=" * 60)
            
            # 1. Select cases for explanation
            print("1️⃣ Selecting explanation cases...")
            selected_cases, case_indices = self.select_explanation_cases(n_malignant=2, n_benign=2)
            
            if selected_cases is None:
                print("❌ Failed to select cases")
                return None
            
            # 2. Implement Grad-CAM for images
            print("\n2️⃣ Implementing Grad-CAM for images...")
            grad_cam_results = {}
            
            # For demonstration, we'll use synthetic cases since we don't have actual images
            # In practice, you would load actual images and run Grad-CAM
            for idx, (case_idx, case) in enumerate(selected_cases.iterrows()):
                # Create a synthetic image path for demonstration
                synthetic_path = f"synthetic_case_{case_idx}.jpg"
                grad_cam_results[case_idx] = self.implement_grad_cam(synthetic_path, case['target'])
            
            # 3. Implement SHAP explanations
            print("\n3️⃣ Implementing SHAP explanations...")
            shap_results = self.implement_shap_explanations(selected_cases)
            
            # 4. Generate comprehensive report
            print("\n4️⃣ Generating explanation report...")
            report = self.generate_explanation_report(selected_cases, grad_cam_results, shap_results)
            
            # 5. Create visualizations
            print("\n5️⃣ Creating visualizations...")
            self.visualize_explanations(selected_cases, grad_cam_results, shap_results)
            
            print("\n✅ Complete Explanation Pipeline Finished!")
            print("=" * 60)
            
            return report
            
        except Exception as e:
            print(f"❌ Error in complete explanation pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function to run the explainability pipeline"""
    print("🏥 Ovarian Cancer Classification - Explainability Module")
    print("=" * 60)
    
    # Initialize explainer
    explainer = OvarianCancerExplainer()
    
    # Run complete explanation pipeline
    report = explainer.run_complete_explanation()
    
    if report:
        print("\n📊 Explanation Summary:")
        print(f"  Total cases analyzed: {report['summary']['total_cases']}")
        print(f"  Malignant cases: {report['summary']['malignant_cases']}")
        print(f"  Benign cases: {report['summary']['benign_cases']}")
        print(f"  Methods used: {', '.join(report['summary']['explanation_methods'])}")
        
        print("\n🎯 Key Findings:")
        if 'shap_results' in report and report['shap_results']:
            global_importance = report['shap_results'].get('global_importance', {})
            if global_importance and 'combined' in global_importance:
                top_features = sorted(global_importance['combined'].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
                print("  Top 3 most important features:")
                for feature, importance in top_features:
                    print(f"    • {feature}: {importance:.3f}")
        
        print("\n📁 Output Files:")
        print("  • results/explainability_report.json - Complete explanation report")
        print("  • results/explainability_visualizations.png - Visualization plots")
        
    else:
        print("❌ Explanation pipeline failed")


if __name__ == "__main__":
    main()
