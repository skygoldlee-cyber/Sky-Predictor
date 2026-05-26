"""
상수 정의 모듈

개선사항:
- 모든 매직 넘버 중앙화
- 타입 힌팅
- 문서화
"""

from enum import Enum
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("skypredictor")
except PackageNotFoundError:
    __version__ = "1.0.0"

VERSION = __version__
APP_NAME = "SkyPredictor"

# ============================================================================
# 데이터 요구사항
# ============================================================================

MIN_MINUTE_BARS_REQUIRED = 20  # 예측에 필요한 최소 분봉 개수

# ============================================================================
# 타임아웃 설정
# ============================================================================

API_TIMEOUT_SECONDS = 30
TICK_SUBSCRIPTION_WAIT_SECONDS = 2
GRACEFUL_SHUTDOWN_TIMEOUT = 5

# ============================================================================
# AI 모델 설정
# ============================================================================

# Claude 모델
CLAUDE_MODEL = "claude-sonnet-4-6"

# Claude 모델 fallback (모델명이 바뀌거나 권한/리전 이슈로 실패할 때 순차 재시도)
CLAUDE_FALLBACK_MODELS = (
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
)

# GPT 모델
GPT_MODEL = "gpt-4o"

# GPT 모델 fallback (모델명이 바뀌거나 권한/리전 이슈로 실패할 때 순차 재시도)
GPT_FALLBACK_MODELS = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
)

# Gemini 모델
# FIX-GEMINI-MODEL: gemini-2.0-flash-lite로 변경.
#   - 무료 티어 RPM이 flash(15)보다 lite(30)가 2배 여유롭다.
#   - 장 중 RPM 소진 → timeout 연쇄 → prov_cooldown 루프를 완화한다.
GEMINI_MODEL = "gemini-2.5-flash"

# FIX-GEMINI-FALLBACK: legacy 1.5 계열 제거.
#   - 운영 로그에서 gemini-1.5-flash가 v1beta 기준 404 NOT_FOUND 반복.
#   - 실제 list_models에 존재하는 2.x 계열만 fallback 체인에 유지.
GEMINI_FALLBACK_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)

# =========================================================================
# LLM output schema (prompt contract)
# =========================================================================

LLM_OUTPUT_SCHEMA: dict = {
    "action": "BUY | SELL | HOLD 중 하나",
    "risk_level": "LOW | MEDIUM | HIGH 중 하나",
    "rationale": "판단 근거 (한국어 1~3문장). OPTIONS_SNAPSHOT이 있으면 PCR/IV Skew/Max Pain을 최소 1개 이상 언급",
    "caution": "무효화 조건/주의사항 (한국어 1문장)",
    "pivot_candidate_probability": "피봇후보 확정 가능성: HIGH | MEDIUM | LOW (후보 없으면 LOW)",
    "pivot_candidate_reason": "피봇후보 확정 가능성 근거 (한국어 1문장, 후보 없으면 '후보 없음')",
}

# =========================================================================
# Runtime limits
# =========================================================================

MAX_FUTURES_TICKS = 100_000

# ============================================================================
# 분석 모드
# ============================================================================

class AnalysisMode(str, Enum):
    """
    분석 모드
    
    SIMPLE: 기본 기술적 지표만 사용
    STANDARD: 기본 옵션 지표 추가 (PCR, Max Pain)
    ADVANCED: 고급 옵션 지표 추가 (Greeks, IV Skew)
    """
    SIMPLE = "simple"
    STANDARD = "standard"
    ADVANCED = "advanced"


# ============================================================================
# 거래 코드
# ============================================================================

class TRCode(str, Enum):
    """
    eBest API 거래 코드
    
    FC0: 선물 실시간 체결
    OC0: 옵션 실시간 체결
    FH0: 선물 실시간 호가 (5단계)
    OH0: 옵션 실시간 호가 (5단계)
    """
    FUTURES = "FC0"
    OPTIONS = "OC0"
    OPTIONS_QUOTE = "OH0"
    FUTURES_BOOK = "FH0"


# ============================================================================
# 예측 방향
# ============================================================================

class PredictionDirection(str, Enum):
    """예측 방향"""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


# ============================================================================
# 로깅 설정
# ============================================================================

DEFAULT_LOG_FILE = "prediction.log"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5

# ============================================================================
# 데이터 보관 기간
# ============================================================================

FUTURES_MINUTE_RETENTION_HOURS = 7  # 선물/KOSPI 분봉 보관 시간 (장 전체 커버: 08:45~15:45)
OPTION_MINUTE_RETENTION_HOURS = 7  # 옵션 분봉 보관 시간

# ============================================================================
# Black-Scholes 설정
# ============================================================================

DEFAULT_RISK_FREE_RATE = 0.03  # 무위험 이자율 3%
MIN_TIME_TO_EXPIRY = 1.0 / 365.0  # 최소 만기일 (1일)
DEFAULT_VOLATILITY = 0.20  # 기본 변동성 20%

# ============================================================================
# API 재시도 설정
# ============================================================================

API_MAX_RETRIES = 2
API_RETRY_DELAY_SECONDS = 2.0
API_BACKOFF_MULTIPLIER = 3.0

# LLM 429 쿨다운 설정
#
# 세 상수는 서로 다른 경로에서 사용된다. 혼용 주의:
#
#   LLM_COOLDOWN_SECONDS_ON_429   (60s)  — 단일 provider 모드 전용.
#       _judge_with_timeout() 에서 429 감지 시 self._llm_rate_limited_until_epoch에 설정.
#       이후 _run_llm_judgment() 상단의 전체 쿨다운 체크가 이 값을 비교한다.
#       dual_llm 모드에서는 이 전체 쿨다운이 적용되는 코드 경로가 없다.
#
#   LLM_PROVIDER_COOLDOWN_ON_429  (120s) — dual_llm 모드 전용.
#       GPT / Gemini 각각 독립적으로 self._provider_rate_limited_until[provider]에 설정.
#       한 provider가 차단되어도 다른 provider로 즉시 fallback 가능.
#       단일 모드에서는 이 값이 사용되지 않는다.
#
#   LLM_PROVIDER_COOLDOWN_ON_TIMEOUT (300s) — timeout 전용 provider 쿨다운.
#       FuturesTimeoutError 발생 시 적용. 429가 아닌 응답 지연이므로
#       LLM_PROVIDER_COOLDOWN_ON_429(120s)보다 길게 설정해 반복 timeout을 방지한다.
#       dual_llm에서 한 provider(예: Gemini)가 계속 timeout되면 이 값만큼 차단되고
#       나머지 provider(GPT)로 자동 fallback된다.
#
# 요약: single → LLM_COOLDOWN_SECONDS_ON_429
#       dual 429 → LLM_PROVIDER_COOLDOWN_ON_429
#       dual timeout → LLM_PROVIDER_COOLDOWN_ON_TIMEOUT
LLM_COOLDOWN_SECONDS_ON_429 = 60.0           # 전체 쿨다운 (단일 provider 모드 전용)
LLM_PROVIDER_COOLDOWN_ON_429 = 120.0         # provider별 429 쿨다운 (dual_llm)
LLM_PROVIDER_COOLDOWN_ON_TIMEOUT = 60.0      # FIX-COOLDOWN: 300→60s. 300s는 장 중 복구가 너무 느렸다.
                                              # timeout 원인이 RPM 제한이면 60s 후 재시도가 적합하다.
LLM_EMPTY_OUTPUT_MAX_RETRIES = 3             # 빈 응답 재시도 횟수 (Gemini 간헐적 현상)

# ============================================================================
# LLM 호출 타이밍 설정 (NW-MNT-01: 분산된 하드코딩 상수 중앙화)
# ============================================================================

# [LLM-FIX-1] timeout 8초 → 15초. Gemini/Claude 응답이 8초를 초과하는 경우가 잦다.
# config.json의 llm_timeout_sec이 우선 적용되므로 여기는 기본값.
LLM_TIMEOUT_SEC = 15.0              # LLM 단일 호출 타임아웃 (초)
LLM_MIN_INTERVAL_SEC = 30.0         # LLM 최소 호출 간격 (초)
LLM_SNAPSHOT_TOLERANCE_SEC = 30.0   # 피드백 스냅샷 허용 오차 (초)

# ============================================================================
# 옵션 구독 기본값
# ============================================================================

DEFAULT_ITM_OPTIONS = 6  # ITM 옵션 개수

# 옵션 피처 세트 버전.
#   v1: 기본 7개 피처 (PCR, IV skew, max pain, ATM microstructure, GEX)
#   v2: v1 + option minute micro-movement 9개 피처
#   v3: v2 + call-put parity divergence 7개 피처 (만기주 전용)
DEFAULT_OPTION_FEATURE_SET = "v5"  # 현재 구현 기준 최신 feature set

# ============================================================================
# 가드레일 독립 제어 플래그 기본값
# (option_feature_set 버전 문자열 비교 대신 이 플래그로 각 가드레일을 개별 제어한다.
#  config.json의 "guardrails" 섹션이 있으면 런타임에 해당 값으로 재정의된다.)
# ============================================================================
GUARDRAIL_OPTION_DEFAULT  = True   # ATM spread / liquidity 기반 (v1~)
GUARDRAIL_BASIS_DEFAULT   = True   # 선물-현물 베이시스 기반 (v1~)
GUARDRAIL_PARITY_DEFAULT  = True   # Call-Put 패리티 이탈 (v3~, 만기주)
GUARDRAIL_BLEED_DEFAULT   = True   # 프리미엄 블리드 (v4~)
GUARDRAIL_OI_DEFAULT      = True   # OI 지지저항 (v5~)

# ============================================================================
# 성능 임계값
# ============================================================================

WARNING_LATENCY_MS = 5000  # 경고 레이턴시 (5초)
ERROR_LATENCY_MS = 10000  # 에러 레이턴시 (10초)
MIN_API_SUCCESS_RATE = 80.0  # 최소 API 성공률 (%)

# ============================================================================
# 차트 시각화 설정
# ============================================================================

CHART_DEFAULT_REFRESH_MS = 500  # 차트 자동 갱신 주기 (ms)
CHART_MAX_BARS = 500  # 렌더 윈도우 상한 (장전체 표시를 위해 증가)

# ============================================================================
# 기술적 지표 설정
# ============================================================================

RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0

# =========================================================================
# TFT (Temporal Fusion Transformer) feature dimensions / horizon
# =========================================================================

FUTURE_KNOWN_DIM = 11
STATIC_DIM = 0
HORIZON_SEC = 300
# PAST_UNKNOWN_DIM: 모델 기본값 파라미터용 참조값.
# 실제 런타임 feature_dim은 pipeline.py에서 config 기반으로 동적 계산된다.
# config.json 기준 현재 구성:
#   OB(10) + CD(8) + OPT_v4(29) + ADAPT(28) + TIME(11) = 86  (multiscale_5m=False)
#   OB(10) + CD(8) + OPT_v4(29) + MS5(8) + ADAPT(28) + TIME(11) = 94  (multiscale_5m=True)
# 아래 값(47)은 OPT_v1 + adaptive 미사용 기준 레거시 값이므로 직접 참조 금지.
PAST_UNKNOWN_DIM = 86  # adaptive=True, OPT_v4, multiscale_5m=False 기준 현행값
