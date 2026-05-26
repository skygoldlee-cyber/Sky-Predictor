"""커맨드라인 인자 파싱 모듈.

main.py에서 분리된 parse_arguments() 단독 모듈.
"""
from __future__ import annotations

import argparse
from typing import Optional

from config import load_config, HORIZON_SEC, DEFAULT_LOG_FILE
from importlib.metadata import version, PackageNotFoundError

try:
    VERSION = version("skypredictor")
except PackageNotFoundError:
    VERSION = "1.0.0"

APP_NAME = "SkyPredictor"


def parse_arguments() -> argparse.Namespace:
    """커맨드라인 인자 파싱.

    MNT-01: 381줄 단일 함수 → 논리 그룹별 주석 섹션으로 구조화.
    추후 _prediction_args(), _adaptive_args(), _ebest_args(), _output_args()로
    분리할 것을 권장한다 (현재는 argparse 단일 parser를 유지하되 섹션을 명확히 함).

    Returns:
        파싱된 인자
    """
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # 기본 실행 (기본값: config.json + eBest live 모드)
    python main.py
    
    # 테스트 모드
    python main.py --test
    
    # Heuristic만 사용 (LLM 비활성화)
    python main.py --heuristic-only --no-ebest-live
    
    # 메트릭 표시
    python main.py --show-metrics
    
    # 디버그 모드
    python main.py --log-level DEBUG
            """
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        help="GUI 대신 CLI 모드로 실행",
    )
    
    # 기본 설정
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="설정 파일 경로 (기본값: config.json)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="로그 레벨 (기본값: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=DEFAULT_LOG_FILE,
        help="로그 파일 경로 (기본값: prediction.log)"
    )
    
    # 예측 설정
    parser.add_argument(
        "--prediction-minutes",
        type=int,
        default=None,
        choices=[5, 10, 30],
        help="예측 시간 (분): 5, 10, 30 중 선택 (기본값: 설정 파일 또는 5)"
    )

    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=None,
        help="BUY 판정 확률 임계값 (0~1). 기본값: config.json 또는 0.62",
    )

    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=None,
        help="SELL 판정 확률 임계값 (0~1). 기본값: config.json 또는 0.38",
    )

    parser.add_argument(
        "--numeric-predictor",
        type=str,
        default=None,
        choices=["transformer", "tft", "combined", "ensemble", "rule_based"],
        help="수치 예측기 모드. 기본값: config.json",
    )

    parser.add_argument(
        "--transformer-weight",
        type=float,
        default=None,
        help="앙상블 내 Transformer 가중치 (0~1). 기본값: config.json 또는 0.5",
    )

    parser.add_argument(
        "--tft-weights-path",
        type=str,
        default=None,
        help="TFT 가중치 경로. 기본값: config.json 또는 predictor 기본값",
    )

    parser.add_argument(
        "--tft-horizon",
        type=int,
        default=None,
        help=f"TFT horizon(seconds/steps). 기본값: config.json 또는 {HORIZON_SEC}",
    )

    parser.add_argument(
        "--disagreement-hold",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="앙상블 disagreement 시 HOLD 강제 여부. 기본값: config.json 또는 True",
    )
    
    parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="LLM 사용 안 함, Heuristic만 사용"
    )
    
    parser.add_argument(
        "--days-to-expiry",
        type=int,
        default=None,
        help="만기일까지 남은 일수 (테스트용, 기본값: 자동 계산)"
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="오더북 시퀀스 길이(초 단위, 1Hz 기준). 기본값: config.json 또는 60",
    )
    parser.add_argument(
        "--fo0-stale-sec",
        type=int,
        default=None,
        help="FO0 미수신 경고 임계값(초). 기본값: config.json 또는 10",
    )
    parser.add_argument(
        "--fo0-log-schema",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="FO0 스키마 키 로그를 활성화",
    )

    parser.add_argument(
        "--preferred-provider",
        type=str,
        default=None,
        choices=["claude", "gpt", "gemini", "openai", "chatgpt"],
        help="LLM 우선 사용 provider 지정 (claude|gpt|gemini). 기본값: config.json 또는 자동",
    )

    parser.add_argument(
        "--dual-llm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="LLM을 gpt/gemini 모두 호출해 결과를 함께 기록/출력 (기본값: config.json)",
    )

    parser.add_argument(
        "--dual-llm-primary-provider",
        type=str,
        default=None,
        choices=["gpt", "gemini", "openai", "chatgpt"],
        help="dual-llm 모드에서 최종 llm_action에 사용할 provider (gpt|gemini)",
    )

    parser.add_argument(
        "--dump-llm-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="LLM user 프롬프트 문자열을 최초 1회 로그로 덤프 (디버그용). 기본값: on (--no-dump-llm-prompt로 비활성화)",
    )

    # Adaptive indicators (optional overrides)
    parser.add_argument(
        "--adaptive-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Adaptive indicators 활성화 여부. 기본값: config.json 또는 True",
    )

    # Adaptive SuperTrend
    parser.add_argument("--ast-atr-min", type=int, default=None, help="Adaptive SuperTrend ATR 최소 기간")
    parser.add_argument("--ast-atr-max", type=int, default=None, help="Adaptive SuperTrend ATR 최대 기간")
    parser.add_argument("--ast-mult-min", type=float, default=None, help="Adaptive SuperTrend multiplier 최소")
    parser.add_argument("--ast-mult-max", type=float, default=None, help="Adaptive SuperTrend multiplier 최대")
    parser.add_argument("--ast-er-period", type=int, default=None, help="Adaptive SuperTrend ER period")
    parser.add_argument("--ast-adx-period", type=int, default=None, help="Adaptive SuperTrend ADX period")
    parser.add_argument(
        "--ast-bb-correction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Adaptive SuperTrend Bollinger width correction 사용 여부",
    )
    parser.add_argument("--ast-bb-period", type=int, default=None, help="Adaptive SuperTrend BB period")
    parser.add_argument("--ast-bb-std", type=float, default=None, help="Adaptive SuperTrend BB std")
    parser.add_argument("--ast-smooth", type=int, default=None, help="Adaptive SuperTrend smoothing period")

    # Adaptive ZigZag
    parser.add_argument("--azz-atr-mult", type=float, default=None, help="Adaptive ZigZag ATR multiplier")
    parser.add_argument("--azz-atr-period", type=int, default=None, help="Adaptive ZigZag ATR period")
    parser.add_argument("--azz-min-thr", type=float, default=None, help="Adaptive ZigZag min threshold (%%)")
    parser.add_argument("--azz-max-thr", type=float, default=None, help="Adaptive ZigZag max threshold (%%)")
    parser.add_argument("--azz-major-ratio", type=float, default=None, help="Adaptive ZigZag major swing ratio")
    parser.add_argument("--azz-max-swings", type=int, default=None, help="Adaptive ZigZag max swings")
    parser.add_argument("--azz-confirm", type=int, default=None, help="Adaptive ZigZag confirmation bars")
    parser.add_argument(
        "--azz-cluster-tol",
        type=float,
        default=None,
        help="Adaptive ZigZag cluster tolerance (%%)",
    )

    parser.add_argument(
        "--azz-struct-lookback",
        type=int,
        default=None,
        help="Adaptive ZigZag structure lookback swings",
    )
    parser.add_argument(
        "--azz-struct-points",
        type=int,
        default=None,
        help="Adaptive ZigZag structure points (high/low samples)",
    )

    parser.add_argument(
        "--azz-freeze-on-confirm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Adaptive ZigZag repainting 완화: confirmation window 동안 후보 스윙(price/idx) 고정 여부",
    )
    
    # 모드 선택
    parser.add_argument(
        "--test",
        action="store_true",
        help="테스트 모드 실행"
    )
    
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="리플레이 모드: 저장된 틱 파일 경로"
    )

    parser.add_argument(
        "--replay-speed",
        type=float,
        default=0.0,
        help="리플레이 속도 배수 (0이면 sleep 없이 최대 속도, 1=실시간, 2=2배속)",
    )

    parser.add_argument(
        "--replay-max-lines",
        type=int,
        default=None,
        help="리플레이 최대 라인 수 (디버깅용)",
    )
    
    # eBest Live 모드
    parser.add_argument(
        "--no-ebest-live",
        action="store_true",
        help="eBest 실시간 모드 비활성화"
    )
    
    parser.add_argument(
        "--duration-sec",
        type=int,
        default=25200,  # 7시간
        help="실시간 모드 실행 시간 (초, 기본값: 25200=7시간)"
    )
    
    # 옵션 설정
    parser.add_argument(
        "--include-options",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="옵션 데이터 포함 여부 (기본값: True, --no-include-options로 비활성화)"
    )
    
    parser.add_argument(
        "--option-month",
        type=str,
        default=None,
        help="옵션 만기월 (YYYYMM, 기본값: 현재/다음 월)"
    )
    
    # 틱 데이터 저장
    parser.add_argument(
        "--out-ticks",
        type=str,
        default=None,
        help="틱 데이터 저장 파일 경로"
    )
    
    parser.add_argument(
        "--no-save-ticks",
        action="store_true",
        help="틱 데이터 저장 안 함"
    )

    parser.add_argument(
        "--compress-ticks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compress tick log output by writing a streaming .jsonl.gz during live capture (기본값: True, --no-compress-ticks로 비활성화)",
    )
    
    # 출력 설정
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="성능 메트릭 표시"
    )
    
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="JSON 형식으로만 출력 (로그 제외)"
    )

    parser.add_argument(
        "--tee",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stdout/stderr를 로그 파일에 tee 할지 여부 (기본값: True, --no-tee로 비활성화)"
    )
    
    return parser.parse_args()


