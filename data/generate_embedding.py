#!/usr/bin/env python3
"""
Generate Embeddings for Wealthy Profiles

This script reads the inputs.json file, loads images for each profile, 
generates embeddings using the same model as the main application, 
and saves the results to wealthy_profiles.json.
"""

import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
import os
from pathlib import Path


def extract_embedding(image_path: str, model, transform, device) -> np.ndarray:
    """Extract embedding from image using the pre-trained model"""
    try:
        # Load and preprocess image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get model output
            outputs = model(image_tensor)
            
            # For ResNet models, the output is the final feature tensor
            # We need to project it to 49 dimensions to match the profiles
            if hasattr(outputs, 'last_hidden_state'):
                # If it has last_hidden_state (unlikely for ResNet), use it
                features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            else:
                # For ResNet, use the direct output
                features = outputs.cpu().numpy()
            
            # Flatten the features
            features = features.flatten()
            
            # Project to 49 dimensions by taking first 49 or sampling
            if len(features) >= 49:
                embedding = features[:49]
            else:
                # If we have fewer than 49 dimensions, pad with zeros
                embedding = np.zeros(49)
                embedding[:len(features)] = features
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def main():
    # Configuration - same as in app/config.py
    MODEL_NAME = "microsoft/resnet-50"
    MAX_IMAGE_SIZE = 224
    DEVICE = "cpu"  # Change to "cuda" if GPU available
    
    print(f"Loading model: {MODEL_NAME}")
    
    # Load pre-trained model and processor
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Model loaded successfully!")
    
    # Read inputs.json
    print("Reading inputs.json...")
    with open('inputs.json', 'r') as f:
        input_data = json.load(f)
    
    print(f"Found {len(input_data['profiles'])} profiles in inputs.json")
    
    # Process each profile and generate embeddings
    processed_profiles = []
    
    for profile in input_data['profiles']:
        name = profile['name']
        net_worth = profile['net_worth']
        occupation = profile['occupation']
        image_path = profile['image']
        
        print(f"\nProcessing {name}...")
        
        if image_path and image_path.strip():
            if os.path.exists(image_path):
                # Generate embedding
                embedding = extract_embedding(image_path, model, transform, DEVICE)
                
                if embedding is not None:
                    processed_profile = {
                        "name": name,
                        "net_worth": net_worth,
                        "occupation": occupation,
                        "embedding": embedding.tolist(),
                        "image": image_path
                    }
                    processed_profiles.append(processed_profile)
                    print(f"  ✓ Generated embedding with {len(embedding)} dimensions")
                else:
                    print(f"  ✗ Failed to generate embedding")
            else:
                print(f"  ✗ Image file not found: {image_path}")
        else:
            print(f"  ✗ No image path provided")
    
    print(f"\nSuccessfully processed {len(processed_profiles)} profiles with embeddings")
    
    # Save the processed profiles to wealthy_profiles.json
    output_data = {
        "profiles": processed_profiles
    }
    
    with open('wealthy_profiles.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(processed_profiles)} profiles to wealthy_profiles.json")
    
    # Show sample of the generated data
    if processed_profiles:
        sample_profile = processed_profiles[0]
        print(f"\nSample profile:")
        print(f"  Name: {sample_profile['name']}")
        print(f"  Net Worth: ${sample_profile['net_worth']:,}")
        print(f"  Occupation: {sample_profile['occupation']}")
        print(f"  Embedding dimensions: {len(sample_profile['embedding'])}")
        print(f"  First 5 embedding values: {sample_profile['embedding'][:5]}")
    
    print("\nDone!")


if __name__ == "__main__":
    main() 