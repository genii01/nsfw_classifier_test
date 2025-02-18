# Chess Piece Classification

체스 말 이미지 분류를 위한 딥러닝 프로젝트입니다. MobileNetV3 아키텍처를 기반으로 하여 5가지 체스 말(Bishop, Knight, Pawn, Queen, Rook)을 분류합니다.

## 프로젝트 구조
```
.
├── config/                  # 설정 파일
│   └── train_config.yaml   # 학습 파라미터 설정
├── data/                   # 데이터 처리 관련
│   ├── dataset.py         # 데이터셋 클래스
│   └── transforms.py      # 데이터 변환 및 증강
├── models/                 # 모델 관련
│   ├── __init__.py
│   ├── model.py           # 모델 아키텍처 정의
│   └── convert.py         # ONNX 변환 유틸리티
├── tools/                  # 유틸리티 스크립트
│   ├── __init__.py
│   └── convert_to_onnx.py # 모델 변환 스크립트
├── utils/                  # 공통 유틸리티
│   └── device.py          # 디바이스 설정
├── dataset/               # 데이터 저장소
│   ├── train/            # 학습 이미지
│   ├── label_mapping.json # 레이블 매핑 정보
│   └── trainset_meta_info.csv # 데이터셋 메타정보
├── saved_models/          # 학습된 모델 저장소
│   ├── best_model.pth    # PyTorch 모델
│   └── model.onnx        # 변환된 ONNX 모델
├── create_dataframe.py    # 데이터셋 메타데이터 생성
├── train.py               # 모델 학습 스크립트
├── Makefile              # 프로젝트 관리
├── pyproject.toml        # 의존성 관리
└── README.md             # 프로젝트 문서
├── app/                    # FastAPI 애플리케이션
│   ├── models/            # 데이터 모델
│   │   └── dto.py        # 데이터 전송 객체 정의
│   ├── services/         # 비즈니스 로직
│   │   └── predictor.py  # 예측 서비스
│   └── main.py           # FastAPI 앱 설정 및 라우터
```

## 주요 기능

### 1. 데이터 처리
- 이미지 데이터셋 구성 및 메타데이터 생성
- 데이터 증강을 통한 모델 일반화:
  - 무작위 회전 (±30도)
  - 무작위 크기 조정 (0.8-1.2배)
  - 색상 변형 (밝기, 대비, 채도, 색조)

### 2. 모델 아키텍처
- MobileNetV3 (Small/Large) 백본
- ImageNet 사전학습 가중치 활용
- 5개 클래스 분류를 위한 커스텀 분류 계층

### 3. 학습 프로세스
- Adam 옵티마이저
- Cross Entropy Loss
- 학습률: 0.001
- 가중치 감쇠: 0.0001
- 조기 종료 (Early Stopping)
- 검증 성능 기반 모델 저장

### 4. 모델 변환 및 검증
#### ONNX 변환
- PyTorch → ONNX 변환 지원
- 동적 배치 크기 처리
- ONNX 런타임 최적화
- 변환 파라미터 커스터마이징:
  ```bash
  poetry run python tools/convert_to_onnx.py \
      --model-path saved_models/best_model.pth \
      --config-path config/train_config.yaml \
      --output-path saved_models/model.onnx \
      --batch-size 1 \
      --img-size 224 \
      --rtol 1e-3 \
      --atol 1e-5 \
      --num-samples 100
  ```

#### 변환 검증
- 자동화된 정확성 검증:
  - PyTorch vs ONNX 출력 비교
  - 랜덤 입력 샘플링 (기본값: 100개)
  - 상대 오차(rtol) 및 절대 오차(atol) 임계값 설정

#### 검증 결과
- JSON 형식의 상세 리포트 생성:
  ```json
  {
    "model_path": "saved_models/best_model.pth",
    "onnx_path": "saved_models/model.onnx",
    "validation_results": {
      "max_difference": 0.000123,
      "mean_difference": 0.000045
    },
    "parameters": {
      "rtol": 1e-3,
      "atol": 1e-5,
      "num_samples": 100
    }
  }
  ```

#### 변환 옵션
| 파라미터 | 설명 | 기본값 |
|----------|------|---------|
| --model-path | PyTorch 모델 경로 (.pth) | required |
| --config-path | 모델 설정 파일 경로 | required |
| --output-path | ONNX 모델 저장 경로 | saved_models/model.onnx |
| --batch-size | 변환 시 배치 크기 | 1 |
| --img-size | 입력 이미지 크기 | 224 |
| --rtol | 상대 오차 임계값 | 1e-3 |
| --atol | 절대 오차 임계값 | 1e-5 |
| --num-samples | 검증 샘플 수 | 100 |

#### 오류 처리
- 변환 실패 시 상세 오류 메시지 제공
- 검증 실패 시 구체적인 차이값 보고
- 임계값 초과 시 예외 발생 및 로깅

#### 사용 예시
1. 기본 설정으로 변환:
   ```bash
   make convert-model
   ```

2. 커스텀 설정으로 변환:
   ```bash
   poetry run python tools/convert_to_onnx.py \
       --model-path saved_models/best_model.pth \
       --config-path config/train_config.yaml \
       --output-path saved_models/custom_model.onnx \
       --batch-size 4 \
       --rtol 1e-4
   ```

3. 검증 결과 확인:
   ```bash
   cat saved_models/conversion_results.json
   ```

#### ONNX 모델 추론
- 최적화된 추론 파이프라인:
  ```python
  from inference.predictor import ChessPredictor

  predictor = ChessPredictor(
      model_path="saved_models/model.onnx",
      label_mapping_path="dataset/label_mapping.json",
      confidence_threshold=0.5
  )

  result = predictor.predict("path/to/image.jpg")
  print(f"Class: {result['label']}")
  print(f"Confidence: {result['confidence']:.4f}")
  ```

- 추론 파이프라인 구성:
  - ImageNet 스타일 전처리
  - 동적 배치 크기 지원
  - CPU/GPU 자동 감지
  - 신뢰도 기반 필터링

- 반환 형식:
  ```python
  {
      "label": "Bishop",          # 예측된 클래스명
      "label_id": 0,              # 클래스 ID
      "confidence": 0.9876        # 예측 신뢰도
  }
  ```

- 오류 처리:
  - 이미지 파일 존재 여부 검증
  - 신뢰도 임계값 기반 unknown 클래스 처리
  - 상세한 예외 메시지

- 커맨드 라인 실행:
  ```bash
  make predict
  ```

#### 추론 설정 옵션
| 파라미터 | 설명 | 기본값 |
|----------|------|---------|
| model_path | ONNX 모델 경로 | required |
| label_mapping_path | 레이블 매핑 파일 경로 | required |
| img_size | 입력 이미지 크기 | 224 |
| confidence_threshold | 신뢰도 임계값 | 0.5 |

#### 이미지 전처리
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

#### 레이블 매핑 형식
```json
{
  "id2label": {
    "0": "Bishop",
    "1": "Knight",
    "2": "Pawn",
    "3": "Queen",
    "4": "Rook"
  }
}
```

## ONNX Batch Inference

### 배치 추론 파이프라인
- 최적화된 배치 처리 파이프라인:
  ```python
  from inference.batch_predictor import ChessBatchPredictor
  
  predictor = ChessBatchPredictor(
      model_path="saved_models/model.onnx",
      label_mapping_path="dataset/label_mapping.json",
      batch_size=32,
      num_workers=4,
      device="cuda" if torch.cuda.is_available() else "cpu"
  )
  
  # 배치 예측 실행
  results = predictor.predict_batch(image_paths)
  ```

### 배치 처리 최적화
#### 메모리 관리
- 동적 배치 크기 조정
  - GPU 메모리 사용량 모니터링
  - OOM (Out of Memory) 방지를 위한 자동 배치 크기 조정
- 메모리 효율적인 데이터 로딩
  - 비동기 데이터 로딩
  - 메모리 매핑된 파일 처리

#### 성능 최적화
- 멀티스레딩 이미지 전처리
- ONNX 런타임 최적화
  - 그래프 최적화
  - 연산자 융합
  - 메모리 재사용
- GPU 가속 지원
  - CUDA 최적화
  - TensorRT 호환성

### 배치 처리 설정 옵션
| 파라미터 | 설명 | 기본값 |
|----------|------|---------|
| model_path | ONNX 모델 경로 | required |
| label_mapping_path | 레이블 매핑 파일 경로 | required |
| batch_size | 배치 크기 | 32 |
| num_workers | 데이터 로딩 워커 수 | 4 |
| device | 실행 디바이스 (cuda/cpu) | auto |
| pin_memory | CUDA 핀 메모리 사용 | True |

### 반환 데이터 형식
```python
[
    {
        "file_name": "image1.jpg",
        "predictions": {
            "label": "Bishop",
            "label_id": 0,
            "confidence": 0.9876
        },
        "processing_time": 0.0234  # 초 단위
    },
    # ... 배치의 다른 이미지들에 대한 결과
]
```

### 성능 메트릭스
- 처리량 (images/second)
- 배치당 평균 처리 시간
- GPU 메모리 사용량
- CPU 사용률

### 오류 처리 및 로깅
```python
try:
    results = predictor.predict_batch(image_paths)
except BatchProcessingError as e:
    logger.error(f"배치 처리 실패: {str(e)}")
    logger.debug(f"실패한 배치: {e.failed_batch}")
    logger.debug(f"스택 트레이스: {e.traceback}")
```

### 커맨드 라인 실행
```bash
# 기본 설정으로 실행
make predict-batch

# 커스텀 설정으로 실행
poetry run python predict_onnx_batch.py \
    --model-path saved_models/model.onnx \
    --label-mapping dataset/label_mapping.json \
    --batch-size 64 \
    --num-workers 8 \
    --device cuda
```

### 성능 최적화 가이드
1. 배치 크기 최적화
   ```python
   # GPU 메모리에 따른 최적 배치 크기 자동 결정
   optimal_batch_size = predictor.find_optimal_batch_size(
       start_size=32,
       max_size=256,
       step=32
   )
   ```

2. 워커 수 최적화
   ```python
   # CPU 코어 수에 기반한 최적 워커 수 설정
   optimal_workers = min(os.cpu_count(), 8)
   ```

3. 메모리 사용량 모니터링
   ```python
   # GPU 메모리 사용량 추적
   predictor.enable_memory_tracking()
   results = predictor.predict_batch(image_paths)
   memory_stats = predictor.get_memory_stats()
   ```

### 벤치마킹 도구
```python
from inference.benchmarking import BatchPredictorBenchmark

benchmark = BatchPredictorBenchmark(predictor)
metrics = benchmark.run(
    test_images=image_paths,
    batch_sizes=[16, 32, 64],
    num_runs=3
)
```

### 벤치마크 결과 예시
```json
{
    "batch_size_32": {
        "avg_throughput": 156.7,
        "avg_latency": 0.204,
        "gpu_memory_used": "1.2GB",
        "cpu_usage": "45%"
    }
}
```

## REST API 서버

### API 서버 실행
```bash
# Poetry로 실행
make serve

# 또는 직접 실행
poetry run python -m app.main
```

### API 엔드포인트

#### 1. 이미지 분류 `/predict`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Request Body**:
  ```
  file: binary (이미지 파일)
  ```
- **Response**:
  ```json
  {
    "label": "Bishop",          // 예측된 클래스
    "label_id": 0,              // 클래스 ID
    "confidence": 0.9876,       // 예측 신뢰도
    "processing_time": 0.0234   // 처리 시간(초)
  }
  ```

#### 2. 헬스 체크 `/health`
- **Method**: GET
- **Response**:
  ```json
  {
    "status": "healthy"
  }
  ```

### API 문서
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 에러 응답
```json
{
  "error": "에러 유형",
  "detail": "상세 에러 메시지"
}
```

### API 사용 예시

#### Python
```python
import requests

# 이미지 파일 업로드
with open("path/to/image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    result = response.json()
    print(f"Predicted class: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
```

#### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/image.jpg"
```

### API 서버 아키텍처

#### 1. 계층 구조
```
app/
├── models/          # 데이터 모델 계층
│   └── dto.py      # - Request/Response DTO
│                   # - 데이터 검증 로직
│
├── services/       # 비즈니스 로직 계층
│   └── predictor.py# - 이미지 처리
│                   # - 모델 추론
│                   # - 결과 포매팅
│
└── main.py         # 애플리케이션 계층
                     # - 라우팅
                     # - 미들웨어
                     # - 에러 처리
```

#### 2. 주요 기능
- **비동기 처리**: FastAPI의 비동기 처리로 높은 처리량
- **자동 문서화**: OpenAPI (Swagger) 문서 자동 생성
- **데이터 검증**: Pydantic 모델을 통한 입출력 검증
- **CORS 지원**: 크로스 오리진 요청 처리
- **에러 처리**: 구조화된 에러 응답

#### 3. 성능 최적화
- 이미지 스트리밍 처리
- 메모리 효율적인 파일 처리
- 비동기 I/O 활용

#### 4. 보안 기능
- 파일 타입 검증
- 요청 크기 제한
- CORS 정책 설정

## 부하 테스트

### Locust 테스트 실행
```bash
# UI 모드로 실행
make load-test-ui

# Headless 모드로 실행 (자동화된 테스트)
make load-test

# 커스텀 설정으로 실행
poetry run python tests/locust/run_load_test.py \
    --users 200 \
    --spawn-rate 20 \
    --run-time 5m \
    --headless
```

### 테스트 시나리오

#### 1. 일반 사용자 시뮬레이션 (ChessClassificationUser)
```python
class ChessClassificationUser(HttpUser):
    wait_time = between(1, 3)  # 1-3초 대기

    @task(1)
    def health_check(self):
        # 헬스 체크 요청

    @task(3)
    def predict_image(self):
        # 이미지 분류 요청
```

#### 2. 고부하 테스트 (ChessClassificationLoadTest)
```python
class ChessClassificationLoadTest(HttpUser):
    wait_time = between(0.1, 0.5)  # 빠른 요청

    @task
    def predict_image_load_test(self):
        # 연속적인 이미지 분류 요청
```

### 테스트 설정 옵션
| 파라미터 | 설명 | 기본값 |
|----------|------|---------|
| --users | 최대 동시 사용자 수 | 100 |
| --spawn-rate | 초당 생성할 사용자 수 | 10 |
| --run-time | 테스트 실행 시간 | 1m |
| --host | 대상 서버 URL | http://localhost:8000 |

### 테스트 리포트
- **UI 대시보드**: http://localhost:8089
- **HTML 리포트**: `load_test_report.html` (headless 모드)

#### 측정 지표
- **응답 시간**
  - 중간값 (Median)
  - 95th 백분위수
  - 99th 백분위수
- **처리량**
  - RPS (Requests Per Second)
  - 초당 실패 수
- **오류율**
  - HTTP 상태 코드별 분포
  - 예외 발생 횟수

### 성능 모니터링
```python
# 테스트 결과 예시
{
    "requests": {
        "total": 15000,
        "success": 14985,
        "failed": 15
    },
    "response_times": {
        "median": 45,
        "95th": 120,
        "99th": 180
    },
    "throughput": {
        "rps": 250.5,
        "failures_per_second": 0.25
    }
}
```

### 부하 테스트 모범 사례
1. **단계적 부하 증가**
   ```bash
   # 점진적으로 부하 증가
   poetry run python tests/locust/run_load_test.py \
       --users 50 --spawn-rate 5 --run-time 2m \
       --headless
   
   poetry run python tests/locust/run_load_test.py \
       --users 100 --spawn-rate 10 --run-time 2m \
       --headless
   ```

2. **임계값 모니터링**
   - 응답 시간: < 200ms
   - 오류율: < 1%
   - CPU 사용률: < 80%
   - 메모리 사용률: < 85%

3. **장기 실행 테스트**
   ```bash
   # 1시간 동안 안정성 테스트
   poetry run python tests/locust/run_load_test.py \
       --users 50 --run-time 1h \
       --headless
   ```

4. **결과 분석**
   ```python
   # 테스트 결과 분석 스크립트
   from locust.stats import RequestStats
   
   def analyze_results():
       stats = RequestStats()
       # 성능 메트릭스 분석
       if stats.get_response_time_percentile(0.95) > 200:
           print("Warning: 95th percentile > 200ms")
   ```

## 설치 및 실행

### 환경 설정
```bash
make setup
```

### 데이터셋 준비
```bash
make create-dataset
```

### 모델 학습
```bash
make train
```

### 모델 변환
```bash
make convert-model
```

### 전체 파이프라인 실행
```bash
make all
```

## 설정 파라미터

### 데이터 설정
```yaml
data:
  train_dir: "dataset/train"
  csv_path: "dataset/trainset_meta_info.csv"
  img_size: 224
  batch_size: 32
  num_workers: 4
  train_split_frac: 0.8
```

### 모델 설정
```yaml
model:
  name: "mobilenet_v3_small"  # or "mobilenet_v3_large"
  num_classes: 5
  pretrained: true
```

### 학습 설정
```yaml
training:
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 5
```

### 데이터 증강 설정
```yaml
augmentation:
  brightness: 0.2
  contrast: 0.2
  saturation: 0.2
  hue: 0.1
  rotation: 30
  scale: [0.8, 1.2]
```

## 의존성

- Python 3.10+
- PyTorch 2.6.0+
- torchvision 0.21.0+
- ONNX 1.15.0+
- ONNX Runtime 1.17.0+
- pandas 2.2.3+
- numpy 1.24.0+
- Pillow 11.1.0+
- OmegaConf 2.3.0+

## 성능 지표

- 학습/검증 손실
- 분류 정확도
- ONNX 변환 정확도:
  - 최대 오차
  - 평균 오차
  - 임계값 기반 검증

## 기여 방법

1. 이 저장소를 포크합니다
2. 새로운 브랜치를 생성합니다
3. 변경사항을 커밋합니다
4. 브랜치에 푸시합니다
5. Pull Request를 생성합니다

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

