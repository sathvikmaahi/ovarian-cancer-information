
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_biomarkers(n_samples, y_target, seed=42):
    """
    Generate synthetic biomarker data for each sample.
    
    Args:
        n_samples: Number of samples
        y_target: Target labels (0=malignant, 1=benign)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic biomarkers
    """
    np.random.seed(seed)
    
    # Age: 28-78 years, uniformly distributed
    ages = np.random.randint(28, 79, n_samples)
    
    # CA-125 Level: Different ranges for benign vs malignant
    ca125_levels = np.zeros(n_samples)
    
    # Benign cases (y=1): 5-35 U/mL
    benign_mask = (y_target == 1)
    ca125_levels[benign_mask] = np.random.uniform(5, 35, np.sum(benign_mask))
    
    # Malignant cases (y=0): 35-600 U/mL (skewed higher)
    malignant_mask = (y_target == 0)
    ca125_levels[malignant_mask] = np.random.gamma(2, 100, np.sum(malignant_mask)) + 35
    ca125_levels[malignant_mask] = np.clip(ca125_levels[malignant_mask], 35, 600)
    
    # BRCA Status: Binary with higher prevalence in malignant cases
    brca_status = np.zeros(n_samples)
    
    # Overall prevalence: 10-15%
    malignant_brca_rate = 0.25  # Higher in malignant cases
    benign_brca_rate = 0.08     # Lower in benign cases
    
    # Assign BRCA status based on target
    brca_status[malignant_mask] = np.random.binomial(1, malignant_brca_rate, np.sum(malignant_mask))
    brca_status[benign_mask] = np.random.binomial(1, benign_brca_rate, np.sum(benign_mask))
    
    # Create DataFrame
    biomarkers = pd.DataFrame({
        'age': ages,
        'ca125_level': ca125_levels,
        'brca_status': brca_status
    })
    
    return biomarkers

def generate_synthetic_images(n_samples, y_target, img_size=(64, 64), seed=42):
    """
    Generate synthetic grayscale images that correlate with the target.
    
    Args:
        n_samples: Number of samples
        y_target: Target labels (0=malignant, 1=benign)
        img_size: Image dimensions (height, width)
        seed: Random seed for reproducibility
    
    Returns:
        Array of synthetic images
    """
    np.random.seed(seed)
    
    images = np.zeros((n_samples, *img_size))
    
    for i in range(n_samples):
        # Base noise
        img = np.random.normal(0.5, 0.1, img_size)
        
        # Add target-specific patterns
        if y_target[i] == 0:  # Malignant - more irregular patterns
            # Add irregular shapes and higher contrast
            for _ in range(3):
                center_x = np.random.randint(10, img_size[1]-10)
                center_y = np.random.randint(10, img_size[0]-10)
                radius = np.random.randint(5, 15)
                
                y_coords, x_coords = np.ogrid[:img_size[0], :img_size[1]]
                mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
                img[mask] += np.random.normal(0.2, 0.1)
            
            # Add some irregular edges
            img += np.random.normal(0, 0.05, img_size)
            
        else:  # Benign - smoother, more regular patterns
            # Add regular circular patterns
            for _ in range(2):
                center_x = np.random.randint(15, img_size[1]-15)
                center_y = np.random.randint(15, img_size[0]-15)
                radius = np.random.randint(8, 20)
                
                y_coords, x_coords = np.ogrid[:img_size[0], :img_size[1]]
                mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
                img[mask] += np.random.normal(0.1, 0.05)
            
            # Smooth the image (simple smoothing without scipy dependency)
            # Apply a simple 3x3 averaging filter
            smoothed = np.copy(img)
            for y in range(1, img_size[0]-1):
                for x in range(1, img_size[1]-1):
                    smoothed[y, x] = np.mean(img[y-1:y+2, x-1:x+2])
            img = smoothed
        
        # Normalize to [0, 1] range
        img = np.clip(img, 0, 1)
        images[i] = img
    
    return images

def main():
    """Main data preparation function"""
    print("Starting data preparation...")
    
    # 1. Load Breast Cancer Wisconsin Dataset
    print("Loading Breast Cancer Wisconsin dataset...")
    cancer_data = load_breast_cancer()
    X = cancer_data.data
    y = cancer_data.target
    feature_names = cancer_data.feature_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Target meaning: 0 = Malignant, 1 = Benign")
    
    # Convert to DataFrame for easier manipulation
    df_features = pd.DataFrame(X, columns=feature_names)
    df_features['target'] = y
    
    # 2. Generate Synthetic Biomarker Data
    print("\nGenerating synthetic biomarkers...")
    biomarkers_df = generate_synthetic_biomarkers(len(y), y)
    
    print("Synthetic biomarkers generated:")
    print(biomarkers_df.head())
    print("\nBiomarker statistics:")
    print(biomarkers_df.describe())
    
    # 3. Generate Synthetic Grayscale Images
    print("\nGenerating synthetic grayscale images...")
    synthetic_images = generate_synthetic_images(len(y), y, img_size=(64, 64))
    print(f"Images generated: {synthetic_images.shape}")
    
    # 4. Combine Features and Create Final Dataset
    print("\nCombining features...")
    final_df = pd.concat([df_features, biomarkers_df], axis=1)
    
    # Add image paths (we'll save images and reference them)
    final_df['image_id'] = [f'image_{i:04d}.npy' for i in range(len(final_df))]
    final_df['target'] = y
    
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Columns: {final_df.columns.tolist()}")
    
    # 5. Create Train/Validation/Test Splits
    print("\nCreating train/validation/test splits...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        final_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print(f"\nTrain target distribution: {np.bincount(y_train)}")
    print(f"Validation target distribution: {np.bincount(y_val)}")
    print(f"Test target distribution: {np.bincount(y_test)}")
    
    # 6. Save Data and Images
    print("\nSaving data and images...")
    # Create directories if they don't exist
    os.makedirs('../data/processed', exist_ok=True)
    os.makedirs('../data/splits', exist_ok=True)
    os.makedirs('../data/images', exist_ok=True)
    
    # Save splits
    X_train.to_csv('../data/splits/train.csv', index=False)
    X_val.to_csv('../data/splits/validation.csv', index=False)
    X_test.to_csv('../data/splits/test.csv', index=False)
    
    # Save images
    for i, img in enumerate(synthetic_images):
        img_path = f'../data/images/image_{i:04d}.npy'
        np.save(img_path, img)
    
    # Save full dataset
    final_df.to_csv('../data/processed/full_dataset.csv', index=False)
    
    # Save image arrays for easy loading
    np.save('../data/processed/all_images.npy', synthetic_images)
    
    print("Data saved successfully!")
    print("Files created:")
    print("- ../data/splits/train.csv")
    print("- ../data/splits/validation.csv")
    print("- ../data/splits/test.csv")
    print("- ../data/processed/full_dataset.csv")
    print("- ../data/processed/all_images.npy")
    print("- ../data/images/image_*.npy (individual image files)")
    
    # 7. Data Summary and Statistics
    print("DATASET SUMMARY: ")
    print(f"Total samples: {len(final_df)}")
    print(f"Features: {len(final_df.columns) - 2}")  # -2 for image_id and target
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Image size: {synthetic_images.shape[1:]} (grayscale)")
    
    print("FEATURE TYPES")
    print(f"Original features (30): {feature_names[:5]} ...")
    print("Synthetic biomarkers (3): age, ca125_level, brca_status")
    print("Image modality: Synthetic grayscale (64x64)")
    
    print("SPLIT SIZES")
    print(f"Train: {len(X_train)} ({len(X_train)/len(final_df)*100:.1f}%)")
    print(f"Validation: {len(X_val)} ({len(X_val)/len(final_df)*100:.1f}%)")
    print(f"Test: {len(X_test)} ({len(X_test)/len(final_df)*100:.1f}%)")
    
    # Save summary to file
    summary = {
        'total_samples': len(final_df),
        'features': len(final_df.columns) - 2,
        'target_distribution': np.bincount(y).tolist(),
        'image_size': list(synthetic_images.shape[1:]),
        'split_sizes': {
            'train': len(X_train),
            'validation': len(X_val),
            'test': len(X_test)
        }
    }
    
    import json
    with open('../data/processed/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nDataset summary saved to ../data/processed/dataset_summary.json")
    
    # 8. Create some visualizations
    print("\nCreating visualizations...")
    
    # Plot some example images
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Synthetic Grayscale Images by Target Class', fontsize=16)
    
    for i in range(4):
        # Malignant examples
        malignant_idx = np.where(y == 0)[0][i]
        axes[0, i].imshow(synthetic_images[malignant_idx], cmap='gray')
        axes[0, i].set_title(f'Malignant (Target=0)')
        axes[0, i].axis('off')
        
        # Benign examples
        benign_idx = np.where(y == 1)[0][i]
        axes[1, i].imshow(synthetic_images[benign_idx], cmap='gray')
        axes[1, i].set_title(f'Benign (Target=1)')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('../data/processed/sample_images.png', dpi=300, bbox_inches='tight')
    print("Sample images saved to ../data/processed/sample_images.png")
    
    # Plot biomarker distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Biomarker Distributions by Target Class', fontsize=16)
    
    for i, col in enumerate(biomarkers_df.columns):
        axes[i].hist(biomarkers_df[y == 0][col], alpha=0.7, label='Malignant', bins=20)
        axes[i].hist(biomarkers_df[y == 1][col], alpha=0.7, label='Benign', bins=20)
        axes[i].set_title(f'{col.replace("_", " ").title()}')
        axes[i].set_xlabel(col.replace("_", " ").title())
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('../data/processed/biomarker_distributions.png', dpi=300, bbox_inches='tight')
    print("Biomarker distributions saved to ../data/processed/biomarker_distributions.png")
    
    print("DATA PREPARATION COMPLETE:")
    print("The dataset is ready for model training!")

if __name__ == "__main__":
    main()
