import argparse
from pathlib import Path
import json
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from models.convert import load_pytorch_model, convert_to_onnx, compare_outputs


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument(
        "--model-path", required=True, help="Path to PyTorch model (.pth)"
    )
    parser.add_argument(
        "--config-path", required=True, help="Path to model config file"
    )
    parser.add_argument(
        "--output-path",
        default="saved_models/model.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for conversion"
    )
    parser.add_argument(
        "--img-size", type=int, default=224, help="Image size for conversion"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for output comparison",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for output comparison",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random samples for validation",
    )

    args = parser.parse_args()

    # 출력 디렉토리 생성
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PyTorch model...")
    pytorch_model = load_pytorch_model(args.model_path, args.config_path)

    print("Converting to ONNX...")
    input_shape = (args.batch_size, 3, args.img_size, args.img_size)
    onnx_path = convert_to_onnx(pytorch_model, args.output_path, input_shape)

    print("Validating conversion...")
    try:
        results = compare_outputs(
            pytorch_model,
            onnx_path,
            input_shape,
            rtol=args.rtol,
            atol=args.atol,
            num_samples=args.num_samples,
        )

        print("\nValidation Results:")
        print(f"Max Difference: {results['max_difference']:.6f}")
        print(f"Mean Difference: {results['mean_difference']:.6f}")

        # 결과를 JSON으로 저장
        results_path = output_dir / "conversion_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "model_path": args.model_path,
                    "onnx_path": args.output_path,
                    "validation_results": results,
                    "parameters": {
                        "rtol": args.rtol,
                        "atol": args.atol,
                        "num_samples": args.num_samples,
                    },
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to {results_path}")
        print(f"ONNX model saved to {onnx_path}")
        print("\nConversion completed successfully!")

    except ValueError as e:
        print(f"\nError during validation: {e}")
        exit(1)


if __name__ == "__main__":
    main()
