#!/usr/bin/env python3
"""
Simple test script for the Wealth Potential Estimator API
"""

import requests
import json
from PIL import Image
import numpy as np
import io

def create_test_image():
    """Create a simple test image"""
    # Create a 224x224 RGB image with random colors
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes

def test_health_endpoint(base_url="http://localhost:8000"):
    """Test the health endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check status: {response.status_code}")
        print(f"Health check response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_predict_endpoint(base_url="http://localhost:8000"):
    """Test the predict endpoint"""
    try:
        # Create test image
        test_image = create_test_image()
        
        # Prepare files for upload
        files = {'file': ('test_image.jpg', test_image, 'image/jpeg')}
        
        # Make request
        response = requests.post(f"{base_url}/predict", files=files)
        
        print(f"Predict endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Prediction successful!")
            print(f"Estimated net worth: ${result['estimated_net_worth']:,}")
            print(f"Confidence score: {result['confidence_score']:.3f}")
            print("Top similar profiles:")
            for profile in result['similar_profiles']:
                print(f"  - {profile['name']}: ${profile['net_worth']:,} (similarity: {profile['similarity_score']:.3f})")
        else:
            print(f"Error response: {response.text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"Predict endpoint test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Wealth Potential Estimator API")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    health_ok = test_health_endpoint()
    
    if not health_ok:
        print("❌ Health check failed. Make sure the API is running.")
        return
    
    # Test predict endpoint
    print("\n2. Testing predict endpoint...")
    predict_ok = test_predict_endpoint()
    
    if predict_ok:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed.")

if __name__ == "__main__":
    main() 