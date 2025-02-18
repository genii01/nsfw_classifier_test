import onnxruntime
import numpy as np
import json
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List, Union, Optional
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class ChessBatchPredictor:
    def __init__(
        self,
        model_path: str,
        label_mapping_path: str,
        img_size: int = 224,
        confidence_threshold: float = 0.5,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        체스 말 분류를 위한 배치 ONNX 모델 추론기

        Args:
            model_path: ONNX 모델 경로
            label_mapping_path: 레이블 매핑 파일 경로
            img_size: 입력 이미지 크기
            confidence_threshold: 신뢰도 임계값
            batch_size: 배치 크기
            num_workers: 이미지 전처리 워커 수
        """
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers

        # ONNX 모델 로드 (CPU/GPU 자동 감지)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)

        # 실제 사용 가능한 provider 확인
        self.device = (
            "GPU" if "CUDAExecutionProvider" in self.session.get_providers() else "CPU"
        )
        print(f"Using {self.device} for inference")

        # 레이블 매핑 로드
        with open(label_mapping_path, "r") as f:
            self.id2label = json.load(f)["id2label"]

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

    def preprocess_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """단일 이미지 전처리"""
        try:
            if isinstance(image_path, str):
                image_path = Path(image_path)

            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                return None

            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
            return image_tensor.numpy()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def _process_batch(
        self, image_paths: List[Union[str, Path]]
    ) -> List[Dict[str, Union[str, int, float]]]:
        """배치 단위 이미지 처리"""
        # 이미지 전처리 (병렬)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            processed_images = list(executor.map(self.preprocess_image, image_paths))

        # 유효한 이미지만 필터링
        valid_images = []
        valid_indices = []
        for idx, img in enumerate(processed_images):
            if img is not None:
                valid_images.append(img)
                valid_indices.append(idx)

        if not valid_images:
            return []

        # 배치 생성
        batch_input = np.stack(valid_images)

        # 모델 입력 이름 가져오기
        input_name = self.session.get_inputs()[0].name

        # 배치 추론 실행
        outputs = self.session.run(None, {input_name: batch_input})
        scores = outputs[0]  # shape: (batch_size, num_classes)

        # 결과 처리
        results = []
        for idx, score in zip(valid_indices, scores):
            # softmax 적용
            exp_scores = np.exp(score - np.max(score))
            probs = exp_scores / exp_scores.sum()

            # 최대 확률 클래스 선택
            max_prob_idx = np.argmax(probs)
            confidence = float(probs[max_prob_idx])

            # 결과 저장
            result = {
                "image_path": str(image_paths[idx]),
                "label": (
                    self.id2label[str(max_prob_idx)]
                    if confidence >= self.confidence_threshold
                    else "unknown"
                ),
                "label_id": (
                    int(max_prob_idx) if confidence >= self.confidence_threshold else -1
                ),
                "confidence": confidence,
            }
            results.append(result)

        return results

    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        save_results: bool = True,
        output_path: Optional[str] = "prediction_results.json",
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        이미지 배치 추론 실행

        Args:
            image_paths: 이미지 경로 리스트
            save_results: 결과를 JSON 파일로 저장할지 여부
            output_path: 결과 저장 경로

        Returns:
            예측 결과 리스트
        """
        all_results = []
        total_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size

        print(f"\nProcessing {len(image_paths)} images in {total_batches} batches...")

        # 배치 단위로 처리
        for i in tqdm(range(0, len(image_paths), self.batch_size)):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_results = self._process_batch(batch_paths)
            all_results.extend(batch_results)

        # 결과 저장
        if save_results and output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                json.dump(
                    {
                        "predictions": all_results,
                        "metadata": {
                            "total_images": len(image_paths),
                            "successful_predictions": len(all_results),
                            "batch_size": self.batch_size,
                            "confidence_threshold": self.confidence_threshold,
                        },
                    },
                    f,
                    indent=2,
                )

            print(f"\nResults saved to {output_file}")

        return all_results


def main():
    """사용 예시"""
    # 예측기 초기화
    predictor = ChessBatchPredictor(
        model_path="saved_models/model.onnx",
        label_mapping_path="dataset/label_mapping.json",
        batch_size=32,
        num_workers=4,
    )

    # 테스트용 이미지 경로 리스트 생성
    image_dir = Path("dataset/train")
    image_paths = list(image_dir.glob("*.jpg"))

    try:
        # 배치 추론 실행
        results = predictor.predict_batch(
            image_paths,
            save_results=True,
            output_path="saved_models/batch_prediction_results.json",
        )

        # 결과 요약 출력
        print("\nPrediction Summary:")
        print(f"Total images processed: {len(image_paths)}")
        print(f"Successful predictions: {len(results)}")

        # 클래스별 통계
        class_counts = {}
        for result in results:
            label = result["label"]
            class_counts[label] = class_counts.get(label, 0) + 1

        print("\nClass Distribution:")
        for label, count in class_counts.items():
            print(f"{label}: {count}")

    except Exception as e:
        print(f"Error during batch prediction: {e}")


if __name__ == "__main__":
    main()
