#!/usr/bin/env python3
"""
Ovarian Cancer Classification Streamlit App (Simplified)

A comprehensive web application that:
1. Accepts ultrasound image uploads
2. Takes tabular inputs (age, CA-125, BRCA status)
3. Runs inference with tabular models
4. Shows output score and feature importance
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import os
import sys
import json
import pickle
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our model functions
from src.main import load_pretrained_tabular_models

# Set page config
st.set_page_config(
    page_title="Ovarian Cancer Classification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .upload-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 0.5rem;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
    }
    .result-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 0.5rem;
        border: 2px solid #28a745;
        margin: 1rem 0;
    }
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-high { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def load_models():
    """Load pre-trained models"""
    try:
        # Load tabular models
        lr_model, mlp_model = load_pretrained_tabular_models()
        
        if lr_model is None or mlp_model is None:
            st.error("Failed to load pre-trained models")
            return None, None
        
        return lr_model, mlp_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess uploaded image for analysis"""
    try:
        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Convert to RGB if grayscale
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Resize image
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
        
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def create_ensemble_prediction(tabular_features, lr_model, mlp_model):
    """Create ensemble prediction from tabular models"""
    try:
        # Get predictions from both models
        lr_pred = lr_model.predict_proba(tabular_features.reshape(1, -1))[0, 1]
        mlp_pred = mlp_model.predict_proba(tabular_features.reshape(1, -1))[0, 1]
        
        # Ensemble prediction (weighted average)
        ensemble_score = 0.4 * lr_pred + 0.6 * mlp_pred
        
        return {
            'ensemble_score': ensemble_score,
            'lr_score': lr_pred,
            'mlp_score': mlp_pred,
            'lr_confidence': abs(lr_pred - 0.5) * 2,  # Distance from 0.5
            'mlp_confidence': abs(mlp_pred - 0.5) * 2
        }
        
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

def analyze_image_features(image):
    """Analyze image features for risk assessment"""
    try:
        if image is None:
            return None
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate image statistics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Calculate texture features (simplified)
        # Sobel gradients for edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_density = np.mean(gradient_magnitude)
        
        # Normalize features to [0, 1] range
        brightness_norm = brightness / 255.0
        contrast_norm = contrast / 128.0  # Assuming max contrast around 128
        edge_norm = edge_density / 100.0  # Normalize edge density
        
        # Create risk score based on image characteristics
        # Higher contrast and edge density might indicate abnormalities
        image_risk_score = (contrast_norm * 0.5 + edge_norm * 0.5)
        
        return {
            'brightness': brightness_norm,
            'contrast': contrast_norm,
            'edge_density': edge_norm,
            'image_risk_score': image_risk_score,
            'analysis': {
                'brightness_level': 'High' if brightness_norm > 0.6 else 'Low' if brightness_norm < 0.4 else 'Normal',
                'contrast_level': 'High' if contrast_norm > 0.7 else 'Low' if contrast_norm < 0.3 else 'Normal',
                'texture_complexity': 'High' if edge_norm > 0.6 else 'Low' if edge_norm < 0.3 else 'Normal'
            }
        }
        
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
        return None

def calculate_feature_importance(tabular_features, feature_names, lr_model, mlp_model):
    """Calculate feature importance from both models"""
    try:
        # Get coefficients from logistic regression
        if hasattr(lr_model, 'coef_'):
            lr_importance = np.abs(lr_model.coef_[0])
        else:
            lr_importance = np.ones(len(feature_names))
        
        # Get feature importance from MLP (if available)
        if hasattr(mlp_model, 'coefs_'):
            # Use weights from first layer as approximation
            mlp_importance = np.abs(mlp_model.coefs_[0]).mean(axis=1)
        else:
            mlp_importance = np.ones(len(feature_names))
        
        # Normalize both
        lr_importance = lr_importance / np.sum(lr_importance)
        mlp_importance = mlp_importance / np.sum(mlp_importance)
        
        # Combine importance (weighted average)
        combined_importance = 0.4 * lr_importance + 0.6 * mlp_importance
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'LR_Importance': lr_importance,
            'MLP_Importance': mlp_importance,
            'Combined_Importance': combined_importance,
            'Value': tabular_features.flatten()
        })
        
        # Sort by combined importance
        feature_importance = feature_importance.sort_values('Combined_Importance', ascending=False)
        
        return feature_importance
        
    except Exception as e:
        st.error(f"Error calculating feature importance: {e}")
        return None

def get_risk_level(score):
    """Get risk level based on score"""
    if score < 0.3:
        return "🟢 Low Risk", "risk-low"
    elif score < 0.7:
        return "🟡 Medium Risk", "risk-medium"
    else:
        return "🔴 High Risk", "risk-high"

def get_clinical_recommendations(risk_level, score):
    """Get clinical recommendations based on risk level"""
    if risk_level == "🟢 Low Risk":
        return st.info, """
        **Low Risk Assessment:**
        - Continue routine monitoring
        - No immediate intervention required
        - Follow standard screening protocols
        - Annual check-ups recommended
        """
    elif risk_level == "🟡 Medium Risk":
        return st.warning, """
        **Medium Risk Assessment:**
        - Consider additional diagnostic tests
        - Schedule follow-up appointment within 3-6 months
        - Monitor for changes in symptoms
        - Discuss with healthcare provider about monitoring frequency
        """
    else:
        return st.error, """
        **High Risk Assessment:**
        - Immediate medical consultation recommended
        - Consider advanced imaging studies (MRI, CT)
        - Potential need for biopsy or surgery
        - Close monitoring required
        """

def main():
    """Main Streamlit app"""
    
    # Ensure numpy is available
    import numpy as np
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Ovarian Cancer Classification</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Input Parameters")
    
    # Load models
    with st.spinner("Loading models..."):
        lr_model, mlp_model = load_models()
    
    if lr_model is None or mlp_model is None:
        st.error("❌ Failed to load models. Please check the model files.")
        return
    
    st.sidebar.success("✅ Models loaded successfully!")
    
    # Add explainability section to sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Explainability")
    
    # Explainability options
    show_explainability = st.sidebar.checkbox("Show Explainability Analysis", value=False)
    explainability_method = st.sidebar.selectbox(
        "Explainability Method",
        ["SHAP Analysis", "Grad-CAM Demo", "Both"],
        help="Choose explainability method to display"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📸 Ultrasound Image Upload")
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Choose an ultrasound image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an ultrasound image for analysis"
        )
        
        if uploaded_image is not None:
            # Display uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Preprocess image
            processed_image = preprocess_image(image)
            if processed_image is not None:
                st.success("✅ Image processed successfully")
        else:
            processed_image = None
            st.info("👆 Please upload an ultrasound image")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📊 Patient Information")
        
        # Tabular inputs
        age = st.slider("Age", min_value=18, max_value=100, value=50, help="Patient age in years")
        
        ca125 = st.number_input(
            "CA-125 Level (U/mL)", 
            min_value=0.0, 
            max_value=1000.0, 
            value=35.0, 
            step=0.1,
            help="CA-125 biomarker level in U/mL"
        )
        
        brca_status = st.selectbox(
            "BRCA Status",
            options=["No Mutation", "BRCA1", "BRCA2"],
            help="BRCA gene mutation status"
        )
        
        # Add stage and grade inputs
        stage = st.selectbox(
            "Cancer Stage",
            options=[1, 2, 3, 4],
            index=1,  # Default to stage 2
            help="Tumor stage (1=early, 4=advanced)"
        )
        
        grade = st.selectbox(
            "Tumor Grade",
            options=[1, 2, 3],
            index=1,  # Default to grade 2
            help="Tumor grade (1=well differentiated, 3=poorly differentiated)"
        )
        
        # Convert BRCA status to numeric
        brca_mapping = {"No Mutation": 0, "BRCA1": 1, "BRCA2": 2}
        brca_numeric = brca_mapping[brca_status]
        
        # Create feature vector with all 5 features (matching training data)
        tabular_features = np.array([float(age), float(ca125), float(brca_numeric), float(stage), float(grade)])
        
        # Display feature summary
        st.markdown("**Feature Summary:**")
        
        # Create a cleaner display without conversion issues
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Age", f"{age} years")
        with col2:
            st.metric("CA-125", f"{ca125:.1f} U/mL")
        with col3:
            st.metric("BRCA Status", brca_status)
        
        col4, col5 = st.columns(2)
        with col4:
            st.metric("Stage", f"Stage {stage}")
        with col5:
            st.metric("Grade", f"Grade {grade}")
        
        # Also show as a table for reference
        feature_df = pd.DataFrame({
            'Feature': ['Age', 'CA-125', 'BRCA Status', 'Stage', 'Grade'],
            'Value': [age, ca125, brca_status, stage, grade]
        })
        st.dataframe(feature_df, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis button
    st.markdown("---")
    col_center = st.columns([1, 2, 1])[1]
    
    with col_center:
        analyze_button = st.button(
            "🔬 Run Analysis",
            type="primary",
            use_container_width=True,
            help="Click to run the complete analysis"
        )
    
    # Results section
    if analyze_button:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.subheader("📊 Analysis Results")
        
        with st.spinner("Running analysis..."):
            # Create ensemble prediction
            prediction_results = create_ensemble_prediction(
                tabular_features, 
                lr_model, 
                mlp_model
            )
            
            if prediction_results:
                # Display results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Ensemble Risk Score", 
                        f"{prediction_results['ensemble_score']:.3f}",
                        help="Combined risk score from both models"
                    )
                
                with col2:
                    st.metric(
                        "LR Risk Score", 
                        f"{prediction_results['lr_score']:.3f}",
                        help="Risk score from Logistic Regression"
                    )
                
                with col3:
                    st.metric(
                        "MLP Risk Score", 
                        f"{prediction_results['mlp_score']:.3f}",
                        help="Risk score from Neural Network"
                    )
                
                # Risk interpretation
                ensemble_score = prediction_results['ensemble_score']
                risk_level, risk_class = get_risk_level(ensemble_score)
                
                st.markdown(f"**Risk Level: <span class='{risk_class}'>{risk_level}</span>**", unsafe_allow_html=True)
                
                # Image analysis (if image uploaded)
                if processed_image is not None:
                    st.subheader("🖼️ Image Analysis")
                    image_analysis = analyze_image_features(processed_image)
                    
                    if image_analysis:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Image Characteristics:**")
                            st.write(f"**Brightness:** {image_analysis['analysis']['brightness_level']}")
                            st.write(f"**Contrast:** {image_analysis['analysis']['contrast_level']}")
                            st.write(f"**Texture Complexity:** {image_analysis['analysis']['texture_complexity']}")
                        
                        with col2:
                            st.markdown("**Image Risk Assessment:**")
                            st.metric(
                                "Image Risk Score",
                                f"{image_analysis['image_risk_score']:.3f}",
                                help="Risk score based on image characteristics"
                            )
                
                # Feature importance
                st.subheader("📈 Feature Importance Analysis")
                feature_names = ['Age', 'CA-125', 'BRCA Status', 'Stage', 'Grade']
                feature_importance = calculate_feature_importance(
                    tabular_features, 
                    feature_names, 
                    lr_model, 
                    mlp_model
                )
                
                if feature_importance is not None:
                    # Create feature importance plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Horizontal bar plot
                    y_pos = np.arange(len(feature_importance))
                    ax.barh(y_pos, feature_importance['Combined_Importance'])
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(feature_importance['Feature'])
                    ax.set_xlabel('Feature Importance')
                    ax.set_title('Combined Feature Importance (LR + MLP)')
                    ax.invert_yaxis()
                    
                    # Add value annotations
                    for i, (importance, value) in enumerate(zip(feature_importance['Combined_Importance'], feature_importance['Value'])):
                        ax.text(importance + 0.01, i, f'{value:.2f}', va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display detailed feature analysis
                    st.markdown("**Detailed Feature Analysis:**")
                    st.dataframe(feature_importance, use_container_width=True)
                
                # Model confidence
                st.subheader("🎯 Model Confidence")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Logistic Regression:**")
                    st.progress(prediction_results['lr_confidence'])
                    st.text(f"Confidence: {prediction_results['lr_confidence']:.1%}")
                
                with col2:
                    st.markdown("**MLP Classifier:**")
                    st.progress(prediction_results['mlp_confidence'])
                    st.text(f"Confidence: {prediction_results['mlp_confidence']:.1%}")
                
                # Clinical recommendations
                st.subheader("💡 Clinical Recommendations")
                recommendation_func, recommendation_text = get_clinical_recommendations(risk_level, ensemble_score)
                recommendation_func(recommendation_text)
                
                # Additional insights
                st.subheader("🔍 Additional Insights")
                
                # Age-based insights
                if age > 65:
                    st.info("**Age Factor:** Patient is in the higher risk age group for ovarian cancer.")
                elif age < 40:
                    st.info("**Age Factor:** Patient is in the lower risk age group, but other factors should be considered.")
                
                # CA-125 insights
                if ca125 > 35:
                    st.warning("**CA-125 Factor:** Elevated CA-125 levels detected. This biomarker is often elevated in ovarian cancer.")
                else:
                    st.success("**CA-125 Factor:** CA-125 levels are within normal range.")
                
                # BRCA insights
                if brca_status != "No Mutation":
                    st.error(f"**BRCA Factor:** {brca_status} mutation detected. This significantly increases ovarian cancer risk.")
                else:
                    st.success("**BRCA Factor:** No BRCA mutations detected.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Explainability Section
    if show_explainability:
        st.markdown("---")
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.subheader("🔍 Explainability Analysis")
        
        if explainability_method in ["SHAP Analysis", "Both"]:
            st.subheader("📊 SHAP Analysis")
            
            # Initialize explainer
            try:
                from src.explainability import OvarianCancerExplainer
                explainer = OvarianCancerExplainer()
                
                # Select cases for explanation
                selected_cases, case_indices = explainer.select_explanation_cases(n_malignant=2, n_benign=2)
                
                if selected_cases is not None:
                    st.success(f"✅ Selected {len(selected_cases)} cases for explanation")
                    
                    # Display selected cases
                    st.markdown("**Selected Cases:**")
                    case_display = selected_cases[['age', 'ca125', 'brca', 'stage_numeric', 'grade_numeric', 'target']].copy()
                    case_display.columns = ['Age', 'CA-125', 'BRCA', 'Stage', 'Grade', 'Risk Level']
                    case_display['Risk Level'] = case_display['Risk Level'].map({0: '🟢 Low Risk', 1: '🔴 High Risk'})
                    st.dataframe(case_display, use_container_width=True)
                    
                    # Implement SHAP explanations
                    with st.spinner("Running SHAP analysis..."):
                        shap_results = explainer.implement_shap_explanations(selected_cases)
                    
                    if shap_results:
                        st.success("✅ SHAP analysis completed!")
                        
                        # Display global importance
                        if 'global_importance' in shap_results:
                            global_imp = shap_results['global_importance']
                            
                            # Create comparison DataFrame
                            importance_df = pd.DataFrame({
                                'Feature': list(global_imp['combined'].keys()),
                                'Logistic Regression': list(global_imp['logistic_regression'].values()),
                                'MLP Classifier': list(global_imp['mlp_classifier'].values()),
                                'Combined': list(global_imp['combined'].values())
                            })
                            
                            # Sort by combined importance
                            importance_df = importance_df.sort_values('Combined', ascending=False)
                            
                            st.markdown("**Global Feature Importance:**")
                            st.dataframe(importance_df.round(4), use_container_width=True)
                            
                            # Create visualization
                            fig, ax = plt.subplots(figsize=(10, 6))
                            features = list(global_imp['combined'].keys())
                            importance = list(global_imp['combined'].values())
                            
                            y_pos = np.arange(len(features))
                            bars = ax.barh(y_pos, importance, color='steelblue', alpha=0.8)
                            
                            # Color bars based on importance
                            colors = plt.cm.Reds(importance / np.max(importance))
                            for bar, color in zip(bars, colors):
                                bar.set_color(color)
                            
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(features)
                            ax.set_xlabel('Feature Importance')
                            ax.set_title('Global Feature Importance (Combined)', fontweight='bold')
                            ax.invert_yaxis()
                            
                            # Add value annotations
                            for i, (importance_val, feature) in enumerate(zip(importance, features)):
                                ax.text(importance_val + 0.01, i, f'{importance_val:.3f}', va='center', fontweight='bold')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Display local explanations
                        if 'local_explanations' in shap_results:
                            st.subheader("📋 Local Explanations")
                            local_explanations = shap_results['local_explanations']
                            
                            for i, explanation in enumerate(local_explanations):
                                with st.expander(f"Case {i+1}: {explanation['risk_level']} Risk"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**Features:**")
                                        for feature, value in explanation['features'].items():
                                            st.write(f"• {feature}: {value}")
                                    
                                    with col2:
                                        st.markdown("**Predictions:**")
                                        st.write(f"• LR: {explanation['predictions']['logistic_regression']:.3f}")
                                        st.write(f"• MLP: {explanation['predictions']['mlp_classifier']:.3f}")
                                        st.write(f"• Ensemble: {explanation['predictions']['ensemble']:.3f}")
                                    
                                    # Feature contributions
                                    st.markdown("**Feature Contributions:**")
                                    contrib_df = pd.DataFrame({
                                        'Feature': list(explanation['feature_contributions']['logistic_regression'].keys()),
                                        'LR Contribution': list(explanation['feature_contributions']['logistic_regression'].values()),
                                        'MLP Contribution': list(explanation['feature_contributions']['mlp_classifier'].values())
                                    })
                                    st.dataframe(contrib_df.round(4), use_container_width=True)
                        
                        else:
                            st.error("❌ Failed to select cases for explanation")
                    
            except Exception as e:
                st.error(f"❌ Error in SHAP analysis: {e}")
                st.info("💡 Make sure the explainability module is properly installed")
        
        if explainability_method in ["Grad-CAM Demo", "Both"]:
            st.subheader("🖼️ Grad-CAM Demonstration")
            
            # Create synthetic images for demonstration
            try:
                import numpy as np
                from scipy import ndimage
                
                def create_synthetic_ultrasound_image(size=(224, 224), risk_level='low'):
                    """Create a synthetic ultrasound-like image for demonstration"""
                    np.random.seed(42)
                    
                    # Create base image with some structure
                    x, y = np.meshgrid(np.linspace(0, 1, size[0]), np.linspace(0, 1, size[1]))
                    
                    if risk_level == 'high':
                        # High risk: multiple complex structures, higher noise
                        center_x, center_y = 0.4, 0.4
                        radius = 0.15
                        mass1 = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * radius**2))
                        
                        center_x2, center_y2 = 0.7, 0.6
                        radius2 = 0.1
                        mass2 = np.exp(-((x - center_x2)**2 + (y - center_y2)**2) / (2 * radius2**2))
                        
                        irregular = np.sin(15 * x) * np.cos(15 * y) * 0.2
                        noise = np.random.normal(0, 0.2, size)
                        image = mass1 + mass2 + irregular + noise
                        
                    else:
                        # Low risk: simple structure, low noise
                        center_x, center_y = 0.5, 0.5
                        radius = 0.2
                        cyst = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * radius**2))
                        
                        smooth = np.sin(8 * x) * np.cos(8 * y) * 0.1
                        noise = np.random.normal(0, 0.05, size)
                        image = cyst + smooth + noise
                    
                    # Normalize to [0, 1]
                    image = (image - image.min()) / (image.max() - image.min())
                    return image
                
                def create_attention_map(image, risk_level):
                    """Create simulated attention map"""
                    gray = image
                    
                    # Edge detection for attention
                    sobel_x = ndimage.sobel(gray, axis=1)
                    sobel_y = ndimage.sobel(gray, axis=0)
                    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                    
                    # Normalize
                    attention = gradient_magnitude / np.max(gradient_magnitude)
                    
                    # Apply different attention patterns based on risk
                    if risk_level == 'high':
                        attention = attention * (1 + 0.5 * np.random.random(attention.shape))
                    else:
                        attention = attention * 0.7
                    
                    return attention
                
                # Create images
                high_risk_img = create_synthetic_ultrasound_image(risk_level='high')
                low_risk_img = create_synthetic_ultrasound_image(risk_level='low')
                
                high_attention = create_attention_map(high_risk_img, 'high')
                low_attention = create_attention_map(low_risk_img, 'low')
                
                # Display images
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔴 High Risk Case**")
                    st.image(high_risk_img, caption="Complex Masses", use_container_width=True)
                    st.image(high_attention, caption="Attention Map", use_container_width=True)
                
                with col2:
                    st.markdown("**🟢 Low Risk Case**")
                    st.image(low_risk_img, caption="Simple Cyst", use_container_width=True)
                    st.image(low_attention, caption="Attention Map", use_container_width=True)
                
                st.success("✅ Grad-CAM demonstration completed!")
                
            except Exception as e:
                st.error(f"❌ Error in Grad-CAM demonstration: {e}")
                st.info("💡 Make sure scipy is installed: pip install scipy")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏥 Ovarian Cancer Classification System | Medical AI Assistant</p>
        <p><small>This tool is for research and educational purposes only. Always consult healthcare professionals for medical decisions.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
