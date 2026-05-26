@echo off
REM 코드 커버리지 리포트 생성 스크립트 (Windows)

echo 코드 커버리지 리포트 생성 중...

REM pytest-cov를 사용하여 커버리지 리포트 생성
pytest tests/ --cov=. --cov-report=html --cov-report=term --cov-report=xml

echo 커버리지 리포트가 htmlcov/ 디렉토리에 생성되었습니다.
echo 터미널에서 커버리지 요약을 확인했습니다.
echo coverage.xml 파일이 생성되었습니다.
