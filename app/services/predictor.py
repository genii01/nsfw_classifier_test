import time
from pathlib import Path
from fastapi import UploadFile
import numpy as np
from PIL import Image
import io
from inference.predictor import ChessPredictor
from app.models.dto import PredictionResponse


class PredictionService:
    def __init__(
        self,
        model_path: str = "saved_models/model.onnx",
        label_mapping_path: str = "dataset/label_mapping.json",
    ):
        self.predictor = ChessPredictor(
            model_path=model_path,
            label_mapping_path=label_mapping_path,
            confidence_threshold=0.5,
        )

    async def predict_image(self, file: UploadFile) -> PredictionResponse:
        start_time = time.time()

        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 이미지 전처리 및 추론
        input_tensor = self.predictor.transform(image).unsqueeze(0).numpy()

        # 모델 입력 이름 가져오기
        input_name = self.predictor.session.get_inputs()[0].name

        # 추론 실행
        outputs = self.predictor.session.run(None, {input_name: input_tensor})
        scores = outputs[0][0]

        # softmax 적용
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # 최대 확률 클래스 선택
        max_prob_idx = np.argmax(probs)
        confidence = float(probs[max_prob_idx])

        # 임계값 검사
        if confidence < self.predictor.confidence_threshold:
            label = "unknown"
            label_id = -1
        else:
            label = self.predictor.id2label[str(max_prob_idx)]
            label_id = int(max_prob_idx)

        processing_time = time.time() - start_time

        return PredictionResponse(
            label=label,
            label_id=label_id,
            confidence=confidence,
            processing_time=processing_time,
        )
