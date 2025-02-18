import subprocess
import argparse
from pathlib import Path


def run_load_test(
    host: str = "http://localhost:8000",
    users: int = 100,
    spawn_rate: int = 10,
    run_time: str = "1m",
    headless: bool = False,
):
    """
    Locust 부하 테스트 실행

    Args:
        host: 대상 서버 URL
        users: 최대 동시 사용자 수
        spawn_rate: 초당 생성할 사용자 수
        run_time: 테스트 실행 시간
        headless: Headless 모드 실행 여부
    """
    locustfile = Path(__file__).parent / "locustfile.py"

    command = [
        "locust",
        "-f",
        str(locustfile),
        "--host",
        host,
    ]

    if headless:
        command.extend(
            [
                "--headless",
                "--users",
                str(users),
                "--spawn-rate",
                str(spawn_rate),
                "--run-time",
                run_time,
                "--html",
                "load_test_report.html",
            ]
        )

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Load test failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run load test for Chess Classification API"
    )
    parser.add_argument(
        "--host", default="http://localhost:8000", help="Target host URL"
    )
    parser.add_argument(
        "--users", type=int, default=100, help="Number of users to simulate"
    )
    parser.add_argument(
        "--spawn-rate", type=int, default=10, help="User spawn rate per second"
    )
    parser.add_argument("--run-time", default="1m", help="Test duration (e.g., 1m, 1h)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")

    args = parser.parse_args()
    run_load_test(
        host=args.host,
        users=args.users,
        spawn_rate=args.spawn_rate,
        run_time=args.run_time,
        headless=args.headless,
    )
