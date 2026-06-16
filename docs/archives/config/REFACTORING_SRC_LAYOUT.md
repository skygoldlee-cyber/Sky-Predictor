# src/ 레이아웃 도입 가이드

## 개요

현재 프로젝트는 루트 디렉토리에 패키지 코드가 직접 배치되어 있습니다. Python 프로젝트 권장사항에 따라 `src/` 레이아웃으로의 이전을 권장합니다.

## 현재 구조

```
SkyPredictor/
├── app/
├── config/
├── core/
├── data/
├── docs/
├── ebestapi/
├── events/
├── gui/
├── indicators/
├── prediction/
├── scripts/
├── services/
├── telegram/
├── tests/
├── trading/
├── main.py
├── pyproject.toml
└── README.md
```

## 권장 구조 (src/ 레이아웃)

```
SkyPredictor/
├── src/
│   └── skypredictor/
│       ├── app/
│       ├── config/
│       ├── core/
│       ├── data/
│       ├── ebestapi/
│       ├── events/
│       ├── gui/
│       ├── indicators/
│       ├── prediction/
│       ├── services/
│       ├── telegram/
│       ├── trading/
│       └── __init__.py
├── tests/
├── scripts/
├── docs/
├── main.py
├── pyproject.toml
└── README.md
```

## 이점

1. **테스트 격리**: 테스트에서 설치된 패키지 대신 로컬 소스를 사용하는 문제 방지
2. **배포 안전성**: 배포 시 불필요한 파일 포함 방지
3. **표준 준수**: Python 프로젝트 권장사항 준수

## 이전 단계

### 1. 디렉토리 구조 변경
```bash
mkdir -p src/skypredictor
mv app config core data ebestapi events gui indicators prediction services telegram trading src/skypredictor/
```

### 2. pyproject.toml 수정
```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["skypredictor*"]
exclude = ["tests*", "docs*", "scripts*"]
```

### 3. import 경로 수정
- `from config import ...` → `from skypredictor.config import ...`
- `from core.utils import ...` → `from skypredictor.core.utils import ...`
- 모든 import 경로를 `skypredictor.` 접두사로 수정

### 4. 테스트 경로 수정
- 테스트 파일의 import 경로도 `skypredictor.` 접두사로 수정

### 5. main.py 수정
- 진입점의 import 경로 수정

## 우선순위

이 리팩토링은 **낮은 우선순위**로 분류됩니다. 현재 프로젝트가 정상적으로 작동하므로, 다음과 같은 경우에 진행을 권장합니다:

- 배포 파이프라인 구축 시
- 테스트 격리 문제 발생 시
- 프로젝트 규모가 커져 구조적 개선이 필요할 때

## 참고

- [Python Packaging Guide](https://packaging.python.org/en/latest/guides/modern-generic-setup/)
- [src Layout vs Flat Layout](https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-src-layout)
