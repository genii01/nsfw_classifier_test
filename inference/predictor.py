import onnxruntime
import numpy as np
import json
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple, Union
import torch
from torchvision import transforms


class ChessPredictor:
    def __init__(
        self,
        model_path: str,
        label_mapping_path: str,
        img_size: int = 224,
        confidence_threshold: float = 0.5,
    ):
        """
        체스 말 분류를 위한 ONNX 모델 추론기

        Args:
            model_path: ONNX 모델 경로
            label_mapping_path: 레이블 매핑 파일 경로
            img_size: 입력 이미지 크기
            confidence_threshold: 신뢰도 임계값
        """
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold

        # ONNX 모델 로드
        self.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        # 레이블 매핑 로드
        with open(label_mapping_path, "r") as f:
            mapping = json.load(f)
            self.id2label = mapping["id2label"]

        # 이미지 전처리 파이프라인
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """이미지를 전처리합니다."""
        if isinstance(image_path, str):
            image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).numpy()

    def predict(self, image_path: Union[str, Path]) -> Dict[str, Union[str, float]]:
        """
        이미지에서 체스 말을 분류합니다.

        Args:
            image_path: 입력 이미지 경로

        Returns:
            예측 결과 딕셔너리:
            {
                'label': 예측된 클래스 레이블,
                'label_id': 예측된 클래스 ID,
                'confidence': 예측 신뢰도
            }
        """
        # 이미지 전처리
        input_tensor = self.preprocess_image(image_path)

        # 모델 입력 이름 가져오기
        input_name = self.session.get_inputs()[0].name

        # 추론 실행
        outputs = self.session.run(None, {input_name: input_tensor})
        scores = outputs[0][0]  # shape: (num_classes,)

        # softmax 적용
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # 최대 확률 클래스 선택
        max_prob_idx = np.argmax(probs)
        confidence = float(probs[max_prob_idx])

        # 임계값 검사
        if confidence < self.confidence_threshold:
            return {"label": "unknown", "label_id": -1, "confidence": confidence}

        # 결과 반환
        return {
            "label": self.id2label[str(max_prob_idx)],
            "label_id": int(max_prob_idx),
            "confidence": confidence,
        }


def main():
    """사용 예시"""
    predictor = ChessPredictor(
        model_path="saved_models/model.onnx",
        label_mapping_path="dataset/label_mapping.json",
        confidence_threshold=0.5,
    )

    # 단일 이미지 추론
    image_path = "dataset/train/bishop_1.jpg"  # 예시 이미지 경로
    try:
        result = predictor.predict(image_path)
        print(f"\nPrediction for {image_path}:")
        print(f"Class: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
