import os
import sys
import time
from pathlib import Path

def main():
    print("="*60)
    print("MULTI-MODAL OVARIAN CANCER CLASSIFICATION PIPELINE")
    print("="*60)
    print("This pipeline will:")
    print("1. Generate synthetic dataset with biomarkers and images")
    print("2. Train three models: Image-only CNN, Tabular-only MLP, Fused Multi-modal")
    print("3. Evaluate and compare all models")
    print("4. Save results and visualizations")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists('src/data_preparation.py'):
        print("Error: Please run this script from the project root directory")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    # Create necessary directories
    print("\nCreating project directories...")
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/splits', exist_ok=True)
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Data Preparation
    print("\n" + "="*50)
    print("STEP 1: DATA PREPARATION")
    print("="*50)
    
    start_time = time.time()
    
    try:
        print("Running data preparation...")
        os.system('python src/data_preparation.py')
        
        # Check if data was created successfully
        if not os.path.exists('data/splits/train.csv'):
            print("Error: Data preparation failed. Check the logs above.")
            sys.exit(1)
            
        print("✓ Data preparation completed successfully!")
        
    except Exception as e:
        print(f"Error during data preparation: {e}")
        sys.exit(1)
    
    data_time = time.time() - start_time
    print(f"Data preparation took: {data_time:.2f} seconds")
    
    # Step 2: Model Training
    print("\n" + "="*50)
    print("STEP 2: MODEL TRAINING")
    print("="*50)
    
    training_start = time.time()
    
    try:
        print("Running model training...")
        os.system('python src/model_implementation.py')
        
        # Check if models were created successfully
        if not os.path.exists('models/image_only_cnn.h5'):
            print("Error: Model training failed. Check the logs above.")
            sys.exit(1)
            
        print("✓ Model training completed successfully!")
        
    except Exception as e:
        print(f"Error during model training: {e}")
        sys.exit(1)
    
    training_time = time.time() - training_start
    total_time = time.time() - start_time
    
    print(f"Model training took: {training_time:.2f} seconds")
    print(f"Total pipeline time: {total_time:.2f} seconds")
    
    # Step 3: Results Summary
    print("\n" + "="*50)
    print("PIPELINE COMPLETION SUMMARY")
    print("="*50)
    
    # Check what was created
    print("Generated files and directories:")
    
    # Data files
    data_files = [
        'data/splits/train.csv',
        'data/splits/validation.csv', 
        'data/splits/test.csv',
        'data/processed/full_dataset.csv',
        'data/processed/all_images.npy',
        'data/processed/dataset_summary.json'
    ]
    
    print("\nData files:")
    for file_path in data_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file_path} ({size:,} bytes)")
        else:
            print(f"  ✗ {file_path} (missing)")
    
    # Model files
    model_files = [
        'models/image_only_cnn.h5',
        'models/tabular_only_mlp.h5',
        'models/fused_multimodal_model.h5'
    ]
    
    print("\nModel files:")
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file_path} ({size:,} bytes)")
        else:
            print(f"  ✗ {file_path} (missing)")
    
    # Result files
    result_files = [
        'results/model_comparison.csv',
        'results/roc_curves.png',
        'results/confusion_matrices.png',
        'results/training_history.png',
        'results/detailed_results.json'
    ]
    
    print("\nResult files:")
    for file_path in result_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file_path} ({size:,} bytes)")
        else:
            print(f"  ✗ {file_path} (missing)")
    
    # Display dataset summary if available
    if os.path.exists('data/processed/dataset_summary.json'):
        print("\nDataset Summary:")
        import json
        with open('data/processed/dataset_summary.json', 'r') as f:
            summary = json.load(f)
        
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Features: {summary['features']}")
        print(f"  Target distribution: {summary['target_distribution']}")
        print(f"  Image size: {summary['image_size']}")
        print(f"  Train/Val/Test split: {summary['split_sizes']['train']}/{summary['split_sizes']['validation']}/{summary['split_sizes']['test']}")
    
    # Display model comparison if available
    if os.path.exists('results/model_comparison.csv'):
        print("\nModel Performance Summary:")
        import pandas as pd
        df = pd.read_csv('results/model_comparison.csv')
        print(df.round(4).to_string(index=False))
    
    print("\n" + "="*50)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Total execution time: {total_time:.2f} seconds")
    print("\nNext steps:")
    print("1. Review the generated visualizations in the 'results' folder")
    print("2. Analyze model performance in 'results/model_comparison.csv'")
    print("3. Use trained models from the 'models' folder for inference")
    print("4. Explore the synthetic dataset in the 'data' folder")
    
    print("\nNote: This is a synthetic dataset for demonstration purposes.")
    print("The models are trained on simulated data, not actual ovarian cancer data.")

if __name__ == "__main__":
    main()
