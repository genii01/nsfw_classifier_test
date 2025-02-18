from pydantic import BaseModel, Field
from typing import Optional
from fastapi import UploadFile, File


class PredictionResponse(BaseModel):
    label: str = Field(..., description="예측된 클래스 레이블")
    label_id: int = Field(..., description="예측된 클래스 ID")
    confidence: float = Field(..., description="예측 신뢰도")
    processing_time: float = Field(..., description="처리 시간 (초)")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 에러 정보")
