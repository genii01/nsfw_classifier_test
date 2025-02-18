from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.services.predictor import PredictionService
from app.models.dto import PredictionResponse, ErrorResponse


app = FastAPI(
    title="Chess Piece Classification API",
    description="체스 말 이미지 분류를 위한 REST API",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 초기화
prediction_service = PredictionService()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_image(file: UploadFile = File(...)):
    """
    이미지를 받아서 체스 말을 분류합니다.

    - **file**: 분류할 이미지 파일 (jpg, png)
    """
    # 파일 형식 검증
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only image files are allowed.",
        )

    try:
        result = await prediction_service.predict_image(file)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.get("/health")
async def health_check():
    """서버 상태를 확인합니다."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
