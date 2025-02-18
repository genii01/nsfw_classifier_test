from inference.predictor import ChessPredictor

predictor = ChessPredictor(
    model_path="saved_models/model.onnx",
    label_mapping_path="dataset/label_mapping.json",
)

result = predictor.predict("./dataset/train/Bishop/00000000_resized.jpg")
print(f"Predicted class: {result['label']}")
print(f"Confidence: {result['confidence']:.4f}")
