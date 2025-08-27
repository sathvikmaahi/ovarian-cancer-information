#!/usr/bin/env python3
"""
Multi-Modal Model Implementation for Ovarian Cancer Classification

This script implements three models:
1. Image-only: Small CNN for image classification
2. Tabular-only: MLP for tabular feature classification
3. Fused model: Multi-modal fusion of image and tabular features

Model Architecture Overview:
- Image encoder: 3 Conv layers + Flatten + Dense head
- Tabular encoder: MLP with ReLU activation
- Fusion: Concatenation + Dense layers + Sigmoid output

Training Strategy:
- Early stopping on validation ROC-AUC
- Evaluation on test set with comprehensive metrics
- Model weight saving for each variant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Deep learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Scikit-learn for traditional ML
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_image_cnn(input_shape=(64, 64, 1)):
    """Create a small CNN for image classification"""
    model = models.Sequential([
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def create_tabular_mlp(input_dim):
    """Create an MLP for tabular feature classification"""
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def create_fused_model(img_shape=(64, 64, 1), tabular_dim=None):
    """Create a fused multi-modal model"""
    
    # Image input and encoder
    img_input = layers.Input(shape=img_shape)
    
    # Image encoder (smaller than standalone)
    img_encoder = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    img_encoder = layers.BatchNormalization()(img_encoder)
    img_encoder = layers.MaxPooling2D((2, 2))(img_encoder)
    img_encoder = layers.Dropout(0.25)(img_encoder)
    
    img_encoder = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(img_encoder)
    img_encoder = layers.BatchNormalization()(img_encoder)
    img_encoder = layers.MaxPooling2D((2, 2))(img_encoder)
    img_encoder = layers.Dropout(0.25)(img_encoder)
    
    img_encoder = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(img_encoder)
    img_encoder = layers.BatchNormalization()(img_encoder)
    img_encoder = layers.GlobalAveragePooling2D()(img_encoder)
    
    # Tabular input and encoder
    tab_input = layers.Input(shape=(tabular_dim,))
    
    tab_encoder = layers.Dense(64, activation='relu')(tab_input)
    tab_encoder = layers.BatchNormalization()(tab_encoder)
    tab_encoder = layers.Dropout(0.3)(tab_encoder)
    
    tab_encoder = layers.Dense(32, activation='relu')(tab_encoder)
    tab_encoder = layers.BatchNormalization()(tab_encoder)
    tab_encoder = layers.Dropout(0.3)(tab_encoder)
    
    # Concatenate features
    concatenated = layers.Concatenate()([img_encoder, tab_encoder])
    
    # Fusion layers
    fusion = layers.Dense(128, activation='relu')(concatenated)
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Dropout(0.5)(fusion)
    
    fusion = layers.Dense(64, activation='relu')(fusion)
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Dropout(0.5)(fusion)
    
    # Output
    output = layers.Dense(1, activation='sigmoid')(fusion)
    
    # Create model
    model = models.Model(inputs=[img_input, tab_input], outputs=output)
    
    return model

def prepare_data(df, images_array):
    """Prepare data for model training"""
    # Extract image IDs and get corresponding images
    image_ids = df['image_id'].values
    image_indices = [int(img_id.split('_')[1].split('.')[0]) for img_id in image_ids]
    images = images_array[image_indices]
    
    # Extract tabular features (exclude image_id and target)
    tabular_features = df.drop(['image_id', 'target'], axis=1).values
    
    # Extract targets
    targets = df['target'].values
    
    return images, tabular_features, targets

def evaluate_model(model, X_img, X_tab, y_true, model_name):
    """Evaluate a model and return metrics"""
    
    # Make predictions
    if model_name == "Image-only CNN":
        y_pred_proba = model.predict(X_img).flatten()
    elif model_name == "Tabular-only MLP":
        y_pred_proba = model.predict(X_tab).flatten()
    else:  # Fused model
        y_pred_proba = model.predict([X_img, X_tab]).flatten()
    
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'confusion_matrix': cm,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }

def main():
    """Main model implementation function"""
    print("Starting model implementation...")
    
    # Load the prepared data
    print("Loading prepared data...")
    train_df = pd.read_csv('../data/splits/train.csv')
    val_df = pd.read_csv('../data/splits/validation.csv')
    test_df = pd.read_csv('../data/splits/test.csv')
    all_images = np.load('../data/processed/all_images.npy')
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Images shape: {all_images.shape}")
    
    # Prepare all datasets
    X_train_img, X_train_tab, y_train = prepare_data(train_df, all_images)
    X_val_img, X_val_tab, y_val = prepare_data(val_df, all_images)
    X_test_img, X_test_tab, y_test = prepare_data(test_df, all_images)
    
    print(f"\nTraining data shapes:")
    print(f"Images: {X_train_img.shape}")
    print(f"Tabular: {X_train_tab.shape}")
    print(f"Targets: {y_train.shape}")
    
    # Normalize tabular features
    scaler = StandardScaler()
    X_train_tab_scaled = scaler.fit_transform(X_train_tab)
    X_val_tab_scaled = scaler.transform(X_val_tab)
    X_test_tab_scaled = scaler.transform(X_test_tab)
    
    # Reshape images for CNN (add channel dimension)
    X_train_img = X_train_img.reshape(-1, 64, 64, 1)
    X_val_img = X_val_img.reshape(-1, 64, 64, 1)
    X_test_img = X_test_img.reshape(-1, 64, 64, 1)
    
    print(f"\nImage shapes after reshaping:")
    print(f"Train: {X_train_img.shape}")
    print(f"Validation: {X_val_img.shape}")
    print(f"Test: {X_test_img.shape}")
    
    # Create directories
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    # Training callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        restore_best_weights=True,
        mode='max'
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    # Model 1: Image-Only CNN
    print("\n" + "="*50)
    print("TRAINING MODEL 1: IMAGE-ONLY CNN")
    print("="*50)
    
    image_cnn = create_image_cnn()
    image_cnn.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    print("Image-only CNN architecture:")
    image_cnn.summary()
    
    # Train the model
    print("\nTraining image-only CNN...")
    image_history = image_cnn.fit(
        X_train_img, y_train,
        validation_data=(X_val_img, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save the model
    image_cnn.save('../models/image_only_cnn.h5')
    print("Image-only CNN saved!")
    
    # Model 2: Tabular-Only MLP
    print("\n" + "="*50)
    print("TRAINING MODEL 2: TABULAR-ONLY MLP")
    print("="*50)
    
    tabular_mlp = create_tabular_mlp(X_train_tab_scaled.shape[1])
    tabular_mlp.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    print("Tabular-only MLP architecture:")
    tabular_mlp.summary()
    
    # Train the model
    print("\nTraining tabular-only MLP...")
    tabular_history = tabular_mlp.fit(
        X_train_tab_scaled, y_train,
        validation_data=(X_val_tab_scaled, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save the model
    tabular_mlp.save('../models/tabular_only_mlp.h5')
    print("Tabular-only MLP saved!")
    
    # Model 3: Fused Multi-Modal Model
    print("\n" + "="*50)
    print("TRAINING MODEL 3: FUSED MULTI-MODAL MODEL")
    print("="*50)
    
    fused_model = create_fused_model(
        img_shape=(64, 64, 1),
        tabular_dim=X_train_tab_scaled.shape[1]
    )
    
    fused_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    print("Fused multi-modal model architecture:")
    fused_model.summary()
    
    # Train the fused model
    print("\nTraining fused multi-modal model...")
    fused_history = fused_model.fit(
        [X_train_img, X_train_tab_scaled], y_train,
        validation_data=([X_val_img, X_val_tab_scaled], y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save the model
    fused_model.save('../models/fused_multimodal_model.h5')
    print("Fused multi-modal model saved!")
    
    # Model Evaluation and Comparison
    print("\n" + "="*50)
    print("MODEL EVALUATION AND COMPARISON")
    print("="*50)
    
    print("Evaluating models on test set...")
    
    results = {}
    results['Image-only CNN'] = evaluate_model(
        image_cnn, X_test_img, None, y_test, "Image-only CNN"
    )
    
    results['Tabular-only MLP'] = evaluate_model(
        tabular_mlp, None, X_test_tab_scaled, y_test, "Tabular-only MLP"
    )
    
    results['Fused Multi-modal'] = evaluate_model(
        fused_model, X_test_img, X_test_tab_scaled, y_test, "Fused Multi-modal"
    )
    
    # Display results
    print("\n=== MODEL PERFORMANCE COMPARISON ===")
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'Precision': [results[model]['precision'] for model in results.keys()],
        'Recall': [results[model]['recall'] for model in results.keys()],
        'F1-Score': [results[model]['f1'] for model in results.keys()],
        'ROC-AUC': [results[model]['roc_auc'] for model in results.keys()]
    })
    
    print(metrics_df.round(4))
    
    # Save results
    metrics_df.to_csv('../results/model_comparison.csv', index=False)
    print("\nResults saved to ../results/model_comparison.csv")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    plt.title('ROC Curves Comparison', fontsize=16)
    for model_name, result in results.items():
        plt.plot(
            result['fpr'], result['tpr'],
            label=f"{model_name} (AUC = {result['roc_auc']:.3f})"
        )
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/roc_curves.png', dpi=300, bbox_inches='tight')
    print("ROC curves saved to ../results/roc_curves.png")
    
    # Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Confusion Matrices Comparison', fontsize=16)
    
    for i, (model_name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'],
            ax=axes[i]
        )
        axes[i].set_title(f'{model_name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('../results/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Confusion matrices saved to ../results/confusion_matrices.png")
    
    # Training History Analysis
    print("\nCreating training history plots...")
    
    # Plot training history for all models
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History Comparison', fontsize=16)
    
    # Loss curves
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].plot(image_history.history['loss'], label='Image CNN')
    axes[0, 0].plot(tabular_history.history['loss'], label='Tabular MLP')
    axes[0, 0].plot(fused_history.history['loss'], label='Fused Model')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].plot(image_history.history['val_loss'], label='Image CNN')
    axes[0, 1].plot(tabular_history.history['val_loss'], label='Tabular MLP')
    axes[0, 1].plot(fused_history.history['val_loss'], label='Fused Model')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_title('Training Accuracy')
    axes[0, 2].plot(image_history.history['accuracy'], label='Image CNN')
    axes[0, 2].plot(tabular_history.history['accuracy'], label='Tabular MLP')
    axes[0, 2].plot(fused_history.history['accuracy'], label='Fused Model')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Validation Accuracy')
    axes[1, 0].plot(image_history.history['val_accuracy'], label='Image CNN')
    axes[1, 0].plot(tabular_history.history['val_accuracy'], label='Tabular MLP')
    axes[1, 0].plot(fused_history.history['val_accuracy'], label='Fused Model')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Validation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Training AUC')
    axes[1, 1].plot(image_history.history['auc'], label='Image CNN')
    axes[1, 1].plot(tabular_history.history['auc'], label='Tabular MLP')
    axes[1, 1].plot(fused_history.history['auc'], label='Fused Model')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].set_title('Validation AUC')
    axes[1, 2].plot(image_history.history['val_auc'], label='Image CNN')
    axes[1, 2].plot(tabular_history.history['val_auc'], label='Tabular MLP')
    axes[1, 2].plot(fused_history.history['val_auc'], label='Fused Model')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Validation AUC')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plots saved to ../results/training_history.png")
    
    # Final summary
    print("\n" + "="*50)
    print("FINAL MODEL SUMMARY")
    print("="*50)
    
    print(f"\nBest performing model by ROC-AUC:")
    best_model = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    print(f"{best_model}: {results[best_model]['roc_auc']:.4f}")
    
    print(f"\nBest performing model by F1-Score:")
    best_f1_model = max(results.keys(), key=lambda x: results[x]['f1'])
    print(f"{best_f1_model}: {results[best_f1_model]['f1']:.4f}")
    
    print(f"\nModel files saved:")
    print("- ../models/image_only_cnn.h5")
    print("- ../models/tabular_only_mlp.h5")
    print("- ../models/fused_multimodal_model.h5")
    
    print(f"\nResults files saved:")
    print("- ../results/model_comparison.csv")
    print("- ../results/roc_curves.png")
    print("- ../results/confusion_matrices.png")
    print("- ../results/training_history.png")
    
    # Save detailed results
    detailed_results = {}
    for model_name, result in results.items():
        detailed_results[model_name] = {
            'metrics': {
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1'],
                'roc_auc': result['roc_auc']
            },
            'predictions': {
                'probabilities': result['y_pred_proba'].tolist(),
                'predictions': result['y_pred'].tolist()
            }
        }
    
    import json
    with open('../results/detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to ../results/detailed_results.json")
    print("\n=== TRAINING COMPLETE ===")
    print("All three models have been trained, evaluated, and saved successfully!")

if __name__ == "__main__":
    main()
