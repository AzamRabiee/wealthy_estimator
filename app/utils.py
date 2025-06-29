import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os
from . import config

class WealthEstimator:
    def __init__(self):
        self.device = torch.device(config.DEVICE)
        self.model_name = config.MODEL_NAME
        
        # Load pre-trained model and processor
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
        # Load wealthy profiles dataset
        self.wealthy_profiles = self._load_wealthy_profiles()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((config.MAX_IMAGE_SIZE, config.MAX_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_wealthy_profiles(self):
        """Load the wealthy profiles dataset"""
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'wealthy_profiles.json')
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data['profiles']
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract embedding from image using the pre-trained model"""
        with torch.no_grad():
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Get model output
            outputs = self.model(image_tensor)
            
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
    
    def compute_similarity(self, user_embedding: np.ndarray) -> list:
        """Compute similarity between user embedding and wealthy profiles"""
        similarities = []
        
        for profile in self.wealthy_profiles:
            profile_embedding = np.array(profile['embedding'])
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                user_embedding.reshape(1, -1), 
                profile_embedding.reshape(1, -1)
            )[0][0]
            
            similarities.append({
                'name': profile['name'],
                'net_worth': profile['net_worth'],
                'occupation': profile['occupation'],
                'similarity_score': float(similarity)
            })
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similarities
    
    def estimate_net_worth(self, similarities: list) -> int:
        """Estimate net worth based on similar profiles"""
        if not similarities:
            return 1000000  # Default estimate
        
        # Weighted average based on similarity scores
        total_weight = 0
        weighted_sum = 0
        
        for profile in similarities[:config.TOP_K_SIMILAR]:
            weight = profile['similarity_score']
            total_weight += weight
            weighted_sum += weight * profile['net_worth']
        
        if total_weight > 0:
            estimated_net_worth = int(weighted_sum / total_weight)
        else:
            estimated_net_worth = 1000000
        
        return estimated_net_worth
    
    def predict(self, image: Image.Image):
        """Main prediction function"""
        # Extract embedding
        user_embedding = self.extract_embedding(image)
        
        # Compute similarities
        similarities = self.compute_similarity(user_embedding)
        
        # Estimate net worth
        estimated_net_worth = self.estimate_net_worth(similarities)
        
        # Get top similar profiles
        top_profiles = similarities[:config.TOP_K_SIMILAR]
        
        # Calculate confidence score (average of top similarities)
        confidence_score = np.mean([p['similarity_score'] for p in top_profiles])
        
        return {
            'estimated_net_worth': estimated_net_worth,
            'similar_profiles': top_profiles,
            'confidence_score': float(confidence_score)
        } 