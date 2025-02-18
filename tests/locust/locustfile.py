import os
from locust import HttpUser, task, between
from pathlib import Path


class ChessClassificationUser(HttpUser):
    wait_time = between(1, 3)  # 요청 사이의 대기 시간 (초)

    def on_start(self):
        """테스트 시작 시 실행되는 설정"""
        # 테스트용 이미지 경로 설정
        self.test_image_path = "dataset/batch_test/00000000_resized.jpg"

        # 이미지 파일 존재 확인
        if not os.path.exists(self.test_image_path):
            raise FileNotFoundError(f"Test image not found: {self.test_image_path}")

        # 헬스 체크
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code != 200:
                raise Exception("API health check failed")

    @task(1)
    def health_check(self):
        """헬스 체크 엔드포인트 테스트"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")
            else:
                response.success()

    @task(3)
    def predict_image(self):
        """이미지 분류 엔드포인트 테스트"""
        try:
            with open(self.test_image_path, "rb") as image_file:
                files = {"file": ("test_image.jpg", image_file, "image/jpeg")}
                with self.client.post(
                    "/predict", files=files, catch_response=True
                ) as response:
                    if response.status_code != 200:
                        response.failure(f"Prediction failed: {response.status_code}")
                    else:
                        result = response.json()
                        if "label" not in result or "confidence" not in result:
                            response.failure("Invalid response format")
                        else:
                            response.success()
        except Exception as e:
            self.environment.runner.log_error(
                request_type="POST", name="/predict", error=str(e)
            )


class ChessClassificationLoadTest(HttpUser):
    """고부하 테스트를 위한 사용자 클래스"""

    wait_time = between(0.1, 0.5)  # 빠른 요청 간격

    def on_start(self):
        self.test_image_path = "./dataset/batch_test/00000000_resized.jpg"
        if not os.path.exists(self.test_image_path):
            raise FileNotFoundError(f"Test image not found: {self.test_image_path}")

    @task
    def predict_image_load_test(self):
        """연속적인 이미지 분류 요청"""
        try:
            with open(self.test_image_path, "rb") as image_file:
                files = {"file": ("test_image.jpg", image_file, "image/jpeg")}
                with self.client.post(
                    "/predict", files=files, catch_response=True
                ) as response:
                    if response.status_code == 200:
                        response.success()
                    else:
                        response.failure(f"Status code: {response.status_code}")
        except Exception as e:
            self.environment.runner.log_error(
                request_type="POST", name="/predict", error=str(e)
            )
