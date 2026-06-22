"""오프라인 검증: 리플레이/검증 세트에서 ``prob``·라벨로 Brier·ECE를 찍는 진입점.

사용 예::

    python scripts/rule_based_backtest_hook.py

실제 틱 배열은 프로젝트 데이터 파이프라인에 맞게 채운 뒤
``prediction.calibration_report.build_validation_report`` 를 호출하면 된다.
"""

from __future__ import annotations

import numpy as np

from prediction.calibration_report import build_validation_report
from prediction.calibration_thresholds import format_tunable_keys_reference


def main() -> None:
    rng = np.random.default_rng(0)
    n = 500
    p = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < p).astype(np.float64)
    print(build_validation_report(p, y, include_tunable_reference=True))


if __name__ == "__main__":
    main()
    print()
    print(format_tunable_keys_reference())
