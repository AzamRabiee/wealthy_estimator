from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
from . import config
from .utils import WealthEstimator
from .models import PredictionResponse, WealthyProfile, ErrorResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION
)

# Initialize the wealth estimator
try:
    wealth_estimator = WealthEstimator()
    logger.info("Wealth estimator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize wealth estimator: {e}")
    wealth_estimator = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Wealth Potential Estimator API",
        "version": config.API_VERSION,
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": wealth_estimator is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_wealth(file: UploadFile = File(...)):
    """
    Predict wealth potential from a selfie image
    
    Args:
        file: Selfie image file (JPEG, PNG, etc.)
    
    Returns:
        PredictionResponse with estimated net worth and similar profiles
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )
        
        # Read and validate image
        try:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Validate image dimensions
            if image.size[0] < 50 or image.size[1] < 50:
                raise HTTPException(
                    status_code=400,
                    detail="Image too small. Minimum size is 50x50 pixels"
                )
                
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Check if model is loaded
        if wealth_estimator is None:
            raise HTTPException(
                status_code=500,
                detail="Model not loaded. Please try again later."
            )
        
        # Make prediction
        logger.info(f"Processing image: {file.filename}")
        result = wealth_estimator.predict(image)
        
        # Convert to response format
        similar_profiles = [
            WealthyProfile(
                name=profile['name'],
                net_worth=profile['net_worth'],
                occupation=profile['occupation'],
                similarity_score=profile['similarity_score']
            )
            for profile in result['similar_profiles']
        ]
        
        response = PredictionResponse(
            estimated_net_worth=result['estimated_net_worth'],
            similar_profiles=similar_profiles,
            confidence_score=result['confidence_score']
        )
        
        logger.info(f"Prediction completed for {file.filename}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred"
        ).__dict__
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 