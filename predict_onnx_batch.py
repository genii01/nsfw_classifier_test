from inference.batch_predictor import ChessBatchPredictor
from pathlib import Path
import os

predictor = ChessBatchPredictor(
    model_path="saved_models/model.onnx",
    label_mapping_path="dataset/label_mapping.json",
    batch_size=32,
)

# 이미지 경로 리스트
image_paths = list(Path("./dataset/batch_test").glob("*.jpg"))
print(image_paths)

# 배치 추론 실행
results = predictor.predict_batch(
    image_paths, save_results=True, output_path="prediction_results.json"
)
