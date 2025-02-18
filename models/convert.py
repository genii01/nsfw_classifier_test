import torch
import onnx
import onnxruntime
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from typing import Tuple, Dict
from .model import create_model


def load_pytorch_model(model_path: str, config_path: str) -> torch.nn.Module:
    """PyTorch 모델을 로드합니다."""
    config = OmegaConf.load(config_path)
    model = create_model(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def convert_to_onnx(
    model: torch.nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    dynamic_batch: bool = True,
) -> str:
    """PyTorch 모델을 ONNX 형식으로 변환합니다."""
    dummy_input = torch.randn(input_shape)

    dynamic_axes = (
        {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        if dynamic_batch
        else None
    )

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    # ONNX 모델 검증
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)

    return save_path


def compare_outputs(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    rtol: float = 1e-3,
    atol: float = 1e-5,
    num_samples: int = 100,
) -> Dict[str, float]:
    """PyTorch와 ONNX 모델의 출력을 비교합니다."""
    ort_session = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    max_diff = 0
    mean_diff = 0

    pytorch_model.eval()

    for i in range(num_samples):
        # 랜덤 입력 생성
        input_tensor = torch.randn(input_shape)

        # PyTorch 추론
        with torch.no_grad():
            pytorch_output = pytorch_model(input_tensor).numpy()

        # ONNX 추론
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]

        # 출력 비교
        diff = np.abs(pytorch_output - ort_output)
        max_diff = max(max_diff, np.max(diff))
        mean_diff = (mean_diff * i + np.mean(diff)) / (i + 1)

        # 허용 오차 검사
        if not np.allclose(pytorch_output, ort_output, rtol=rtol, atol=atol):
            raise ValueError(
                f"Output mismatch detected at sample {i}. "
                f"Max difference: {np.max(diff)}, Mean difference: {np.mean(diff)}"
            )

    return {"max_difference": float(max_diff), "mean_difference": float(mean_diff)}
