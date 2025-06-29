---
title: Wealth Potential Estimator
emoji: 💰
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: 1.0.0
app_file: app/main.py
pinned: false
---

# Wealth Potential Estimator API

A machine learning service that estimates a user's potential net worth based on a submitted selfie image. The API analyzes facial features and compares them to a database of wealthy individuals to provide wealth potential estimates.

# Live URL of the Endpoint
* Repo (Github): https://github.com/AzamRabiee/wealthy_estimator
* Repo (HF): https://huggingface.co/spaces/Azam-Rabiee/wealthy-estimator-api/tree/main
* Swagger: https://azam-rabiee-wealthy-estimator-api.hf.space/docs

## 🏗️ Architecture

### Overview
This project implements a **Wealth Potential Estimator** using:
- **FastAPI** for the REST API framework
- **Hugging Face Transformers** for pre-trained vision models
- **PyTorch** for deep learning inference
- **Docker** for containerization and deployment

### Key Components

1. **Vision Model**: Uses `microsoft/resnet-50` for feature extraction
2. **Embedding Extraction**: Converts selfie images to high-dimensional feature vectors
3. **Similarity Computation**: Uses cosine similarity to find similar wealthy profiles
4. **Wealth Estimation**: Weighted average based on similarity scores

### Data Flow
```
Selfie Image → Preprocessing → Feature Extraction → Similarity Search → Wealth Estimation → API Response
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.9+ (for local development)

### Using Docker (Recommended)

1. **Clone and build**:
```bash
git clone <repository-url>
cd wealth-estimator-api
docker-compose up --build
```

2. **Access the API**:
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Predict Endpoint: http://localhost:8000/predict

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the application**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

### POST `/predict`
Predicts wealth potential from a selfie image.

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file (JPEG, PNG, etc.)

**Response**:
```json
{
  "estimated_net_worth": 50000000,
  "currency": "USD",
  "similar_profiles": [
    {
      "name": "Elon Musk",
      "net_worth": 250000000000,
      "occupation": "Entrepreneur",
      "similarity_score": 0.85
    }
  ],
  "confidence_score": 0.75
}
```

### GET `/health`
Health check endpoint.

### GET `/`
Root endpoint with API information.

## 🔧 Configuration

Key configuration options in `app/config.py`:

- `MODEL_NAME`: Pre-trained model to use (default: "microsoft/resnet-50")
- `MAX_IMAGE_SIZE`: Input image size (default: 224)
- `DEVICE`: Computation device (default: "cpu")
- `TOP_K_SIMILAR`: Number of similar profiles to return (default: 3)

## 🧠 Model Details

### Model Selection
- **Model**: `microsoft/resnet-50`
- **Reasoning**: 
  - Pre-trained on ImageNet with strong feature extraction capabilities
  - Lightweight and fast for inference
  - Good balance between accuracy and performance

### Feature Extraction Process
1. **Image Preprocessing**: Resize to 224x224, normalize with ImageNet stats
2. **Feature Extraction**: Use ResNet-50's last hidden state
3. **Embedding Normalization**: L2 normalization for consistent similarity computation

### Similarity Computation
- **Metric**: Cosine Similarity
- **Reasoning**: 
  - Scale-invariant similarity measure
  - Works well with normalized embeddings
  - Computationally efficient

### Wealth Estimation Algorithm
```python
estimated_net_worth = Σ(similarity_score × profile_net_worth) / Σ(similarity_scores)
```

## 📊 Dataset

The wealthy profiles dataset (`data/wealthy_profiles.json`) contains:
- 10 mock wealthy individuals
- Each profile includes: name, net worth, occupation, and pre-computed embeddings
- Net worth values range from $65B to $250B

**Note**: This is a demonstration dataset. In production, you would:
- Use real wealthy individual data
- Implement proper data privacy measures
- Use more sophisticated embedding generation

## 🏗️ Engineering Decisions

### Framework Choices
- **FastAPI**: Modern, fast, automatic API documentation
- **PyTorch**: Industry standard for deep learning
- **Transformers**: Easy access to pre-trained models

### Performance Optimizations
- Model loading at startup (not per request)
- Image preprocessing optimization
- Efficient similarity computation

### Error Handling
- Comprehensive input validation
- Graceful error responses
- Detailed logging

### Security Considerations
- Input file validation
- Non-root Docker user
- Health checks for monitoring

## 🧪 Testing

### Manual Testing
1. Use the interactive API docs at `/docs`
2. Upload a selfie image
3. Verify the response format and values

### Example cURL Request
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_selfie.jpg"
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d
```

### Cloud Deployment Options
- **Heroku**: Use the provided Dockerfile
- **AWS ECS**: Deploy using docker-compose
- **Google Cloud Run**: Container-based deployment
- **Azure Container Instances**: Serverless container deployment

## 📝 Assumptions

1. **Model Accuracy**: This is a demonstration project - accuracy is secondary to robustness
2. **Wealth Correlation**: Assumes facial features correlate with wealth potential (for demo purposes)
3. **Dataset**: Uses mock wealthy profiles - not real data
4. **Privacy**: No actual personal data is stored or processed beyond the request
5. **Scalability**: Designed for moderate traffic - can be scaled with load balancers

## 🔮 Future Improvements

1. **Model Enhancement**: 
   - Fine-tune on wealth-related datasets
   - Use ensemble models for better accuracy
   
2. **Data Pipeline**:
   - Real-time wealthy profile updates
   - More diverse dataset
   
3. **API Features**:
   - Batch processing
   - Authentication and rate limiting
   - Caching for similar requests
   
4. **Monitoring**:
   - Request/response logging
   - Performance metrics
   - Error tracking

## 📄 License

This project is for demonstration purposes only.

## 🤝 Contributing

This is a demonstration project for a coding interview. For production use, please implement proper security, privacy, and ethical considerations. 