#!/usr/bin/env python3
"""
Test script for Streamlit app functionality
"""

import numpy as np
import pandas as pd
from PIL import Image
import cv2

# Test the functions from the Streamlit app
def test_image_preprocessing():
    """Test image preprocessing function"""
    print("Testing image preprocessing...")
    
    # Create a test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_pil = Image.fromarray(test_image)
    
    # Import the function
    from streamlit_app_simple import preprocess_image
    processed = preprocess_image(test_pil)
    
    if processed is not None:
        print(f"✅ Image preprocessing successful: {processed.shape}")
        return True
    else:
        print("❌ Image preprocessing failed")
        return False

def test_ensemble_prediction():
    """Test ensemble prediction function"""
    print("Testing ensemble prediction...")
    
    # Create test features
    test_features = np.array([50, 35.0, 0, 2, 2])  # age, ca125, brca, stage, grade
    
    # Import the function
    from streamlit_app_simple import create_ensemble_prediction
    
    # Mock models (we'll just test the function structure)
    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])
    
    mock_lr = MockModel()
    mock_mlp = MockModel()
    
    try:
        result = create_ensemble_prediction(test_features, mock_lr, mock_mlp)
        if result:
            print(f"✅ Ensemble prediction successful: {result}")
            return True
        else:
            print("❌ Ensemble prediction failed")
            return False
    except Exception as e:
        print(f"❌ Ensemble prediction error: {e}")
        return False

def test_feature_importance():
    """Test feature importance calculation"""
    print("Testing feature importance calculation...")
    
    # Create test features
    test_features = np.array([50, 35.0, 0, 2, 2])  # age, ca125, brca, stage, grade
    feature_names = ['Age', 'CA-125', 'BRCA Status', 'Stage', 'Grade']
    
    # Import the function
    from streamlit_app_simple import calculate_feature_importance
    
    # Mock models
    class MockLRModel:
        def __init__(self):
            self.coef_ = np.array([[0.3, 0.25, 0.2, 0.15, 0.1]])  # 5 features
    
    class MockMLPModel:
        def __init__(self):
            self.coefs_ = [np.random.rand(5, 5)]  # 5 input features
    
    mock_lr = MockLRModel()
    mock_mlp = MockMLPModel()
    
    try:
        result = calculate_feature_importance(test_features, feature_names, mock_lr, mock_mlp)
        if result is not None:
            print(f"✅ Feature importance calculation successful")
            print(result)
            return True
        else:
            print("❌ Feature importance calculation failed")
            return False
    except Exception as e:
        print(f"❌ Feature importance calculation error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Streamlit App Functions")
    print("=" * 50)
    
    tests = [
        test_image_preprocessing,
        test_ensemble_prediction,
        test_feature_importance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Streamlit app is ready to run.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
