"""Numeric predictor component for the pipeline.

If transformer weights are available, this predictor runs torch inference.
Otherwise it falls back to the existing rule-based logic.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Protocol

import numpy as np

from config import FUTURE_KNOWN_DIM, HORIZON_SEC, PAST_UNKNOWN_DIM
from .features.features import CD_KEYS, OB_KEYS


logger = logging.getLogger(__name__)


class AdaptiveEnsembleWeightTracker:
    """Transformer/TFT 예측 정확도를 추적하고 앙상블 가중치를 동적으로 조정한다.

    NW-CON-01: update/get_weights/reset 은 피드백 루프 스레드와 예측 스레드에서
    동시에 호출될 수 있으므로 _lock으로 deque 접근을 직렬화한다.

    [IMP-ENS-01] Brier Score 기반 점수화:
        이진(0/1) 정오 집계 대신 예측 확률과 실제 결과 간 Brier Score를 사용한다.
        prob=0.51 맞춤과 prob=0.88 맞춤을 동일하게 취급하는 편향을 제거한다.

    [IMP-ENS-02] 동적 창 크기:
        시장 레짐에 따라 window를 동적으로 조정한다.
        set_window()를 통해 파이프라인이 레짐 변화 시 창 크기를 갱신한다.
        - RANGE / WEAK 레짐: window=30 (안정적 시장, 긴 이력)
        - STRONG_TREND: window=12 (추세 전환 빠름)
        - 만기주(dte_weight_norm > 0.5): window=8
    """

    def __init__(self, window: int = 20):
        self._lock = threading.Lock()  # NW-CON-01
        self._window = max(4, int(window))
        self._transformer_scores: deque = deque(maxlen=self._window)
        self._tft_scores: deque = deque(maxlen=self._window)
        self._transformer_w: deque = deque(maxlen=self._window)
        self._tft_w: deque = deque(maxlen=self._window)

    def set_window(self, window: int) -> None:
        """창 크기를 동적으로 변경한다. 기존 데이터는 최신 window개만 유지한다."""
        new_win = max(4, int(window))
        with self._lock:
            if new_win == self._window:
                return
            self._window = new_win
            # 기존 deque를 새 크기로 재생성 (최신 데이터 유지)
            self._transformer_scores = deque(list(self._transformer_scores)[-new_win:], maxlen=new_win)
            self._tft_scores         = deque(list(self._tft_scores)[-new_win:],         maxlen=new_win)
            self._transformer_w      = deque(list(self._transformer_w)[-new_win:],      maxlen=new_win)
            self._tft_w              = deque(list(self._tft_w)[-new_win:],              maxlen=new_win)

    def update(
        self,
        *,
        transformer_correct: bool,
        tft_correct: bool,
        transformer_weight: float = 1.0,
        tft_weight: float = 1.0,
        transformer_prob: Optional[float] = None,
        tft_prob: Optional[float] = None,
    ) -> None:
        """피드백 결과를 기록한다.

        [IMP-ENS-01] transformer_prob / tft_prob가 제공되면
        Brier Score 기반 점수(1 - (prob - actual)^2)를 사용한다.
        미제공 시 기존 이진 점수(0/1)로 폴백한다.

        Args:
            transformer_correct: Transformer 예측이 실제로 맞았는지 여부.
            tft_correct: TFT 예측이 실제로 맞았는지 여부.
            transformer_weight: Transformer 가중치 스케일 (기본 1.0).
            tft_weight: TFT 가중치 스케일 (기본 1.0).
            transformer_prob: Transformer 예측 확률 [0,1]. None이면 이진 점수 사용.
            tft_prob: TFT 예측 확률 [0,1]. None이면 이진 점수 사용.
        """
        tw = max(0.0, float(transformer_weight))
        fw = max(0.0, float(tft_weight))

        # [IMP-ENS-01] Brier Score: score = 1 - (prob - actual)^2
        # actual=1(정답), actual=0(오답). prob이 없으면 이진 점수.
        def _brier_score(prob: Optional[float], correct: bool) -> float:
            if prob is None:
                return 1.0 if correct else 0.0
            try:
                p = max(0.0, min(1.0, float(prob)))
                actual = 1.0 if correct else 0.0
                return float(1.0 - (p - actual) ** 2)
            except Exception:
                return 1.0 if correct else 0.0

        t_score = _brier_score(transformer_prob, bool(transformer_correct))
        f_score = _brier_score(tft_prob, bool(tft_correct))

        with self._lock:
            self._transformer_scores.append(t_score * tw)
            self._tft_scores.append(f_score * fw)
            self._transformer_w.append(tw)
            self._tft_w.append(fw)

    def get_weights(self) -> tuple[float, float]:
        """현재 누적 Brier Score 기반 (transformer_weight, tft_weight) 튜플을 반환한다.

        데이터 부족 시 (0.5, 0.5)를 반환한다.
        """
        with self._lock:
            t_den = float(max(1e-9, sum(self._transformer_w) if self._transformer_w else max(1, len(self._transformer_scores))))
            f_den = float(max(1e-9, sum(self._tft_w) if self._tft_w else max(1, len(self._tft_scores))))
            t_acc = float(sum(self._transformer_scores)) / t_den
            f_acc = float(sum(self._tft_scores)) / f_den
        total = t_acc + f_acc
        if total < 1e-9:
            return 0.5, 0.5
        return float(t_acc / total), float(f_acc / total)

    def reset(self) -> None:
        """누적 통계를 초기화한다."""
        with self._lock:
            self._transformer_scores.clear()
            self._tft_scores.clear()
            self._transformer_w.clear()
            self._tft_w.clear()


_DEFAULT_WEIGHTS = "prediction/weights/transformer_5m.pt"
_DEFAULT_TFT_WEIGHTS = "prediction/weights/tft_5m.pt"


@dataclass
class TransformerPredictionResult:
    """Numeric prediction result.

    Attributes:
        prob: Up probability in [0, 1].
        signal: BUY/SELL/HOLD.
        confidence: HIGH/MEDIUM/LOW.
        feature_snapshot: Features used/attached for downstream context.
    """
    prob: float
    signal: str
    confidence: str
    feature_snapshot: Dict[str, Any]


@dataclass
class ModelInput:
    sequence: Optional[np.ndarray] = None
    past_known: Optional[np.ndarray] = None
    future_known: Optional[np.ndarray] = None
    feature_snapshot: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    schema_version: str = ""

    def __post_init__(self) -> None:
        if self.feature_snapshot is None:
            self.feature_snapshot = {}
        if self.meta is None:
            self.meta = {}


class PredictionResult(Protocol):
    prob: float
    signal: str
    confidence: str
    feature_snapshot: Dict[str, Any]


def _classify(
    *,
    prob: float,
    spread: float,
    buy_threshold: float,
    sell_threshold: float,
    confidence_high_margin: float = 0.15,
    confidence_mid_margin: float = 0.08,
    confidence_spread_max_for_high: float = 1.0,
) -> tuple[str, str]:
    p = float(prob)
    bt = float(buy_threshold)
    st = float(sell_threshold)

    if p >= bt:
        signal = "BUY"
    elif p <= st:
        signal = "SELL"
    else:
        signal = "HOLD"

    margin = abs(p - 0.5)
    if margin >= float(confidence_high_margin) and float(spread or 0.0) <= float(confidence_spread_max_for_high):
        confidence = "HIGH"
    elif margin >= float(confidence_mid_margin):
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    return signal, confidence


def _max_pairwise_abs_diff(probs: List[float]) -> float:
    """모델별 확률 리스트에서 최대 쌍별 절대 차이 (연속 불일치 지표)."""
    if len(probs) < 2:
        return 0.0
    m = 0.0
    for a, b in combinations(probs, 2):
        try:
            d = abs(float(a) - float(b))
        except Exception:
            continue
        if d > m:
            m = d
    return float(m)


def adjust_confidence_by_conformal_interval_width(
    confidence: str,
    prob_lower: Optional[float],
    prob_upper: Optional[float],
    *,
    width_max_for_high: float = 0.35,
    width_max_for_medium: float = 0.55,
) -> str:
    """Conformal 예측 구간 폭이 클수록 캘리브레이션 불확실성이 크다고 보고 신뢰도를 낮춘다.

    - `prob_lower`/`prob_upper`가 없으면 변경 없음.
    - 폭 >= ``width_max_for_medium`` → ``LOW``.
    - 그렇지 않고 폭 >= ``width_max_for_high`` 이고 기존이 ``HIGH`` → ``MEDIUM``.

    검증 세트에서 Brier/ECE를 측정해 두 임계값을 조정할 수 있다.
    """
    if prob_lower is None or prob_upper is None:
        return confidence
    try:
        w = float(prob_upper) - float(prob_lower)
    except Exception:
        return confidence
    w = max(0.0, min(1.0, w))
    try:
        wh = float(width_max_for_high)
        wm = float(width_max_for_medium)
    except Exception:
        return confidence
    if wm < wh:
        wm = wh
    c = str(confidence or "LOW").strip().upper()
    if c not in ("HIGH", "MEDIUM", "LOW"):
        c = "LOW"
    if w >= wm:
        return "LOW"
    if w >= wh and c == "HIGH":
        return "MEDIUM"
    return c


def _merge_rule_based_weights(raw: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """config ``rule_based_weights`` 병합. 미지정 키는 기본값."""
    defaults: Dict[str, float] = {
        "w_obi": 0.55,
        "w_lvl1": 0.30,
        "w_slope": 0.15,
        "mom_scale": 50.0,
        "mom_coef": 0.10,
        "pressure_clip": 0.48,
        "spread_penalty_coef": 0.25,
    }
    if not isinstance(raw, Mapping):
        return dict(defaults)
    out = dict(defaults)
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            pass
    return out


def compute_rule_based_probability(
    snap: Dict[str, Any],
    sequence: Optional[np.ndarray],
    *,
    weights: Mapping[str, float],
    mom_multiplier: float = 1.0,
    confidence_spread_max_for_high: float = 1.0,
) -> tuple[float, float]:
    """호가·캔들 기반 휴리스틱 확률과 스프레드(pt). ``TransformerPredictor._rule_based`` 와 동일 식."""
    def _sf(key: str, default: float = 0.0) -> float:
        try:
            v = snap.get(key)
            return float(default) if v is None else float(v)
        except Exception:
            return float(default)

    w = dict(weights)
    obi = _sf("obi", 0.0)
    spread = _sf("spread", 0.0)
    lvl1 = _sf("level1_ratio", 0.0)
    bid_slope = _sf("bid_slope", 0.0)
    offer_slope = _sf("offer_slope", 0.0)

    ret3 = 0.0
    vol_accel = 0.0
    try:
        if sequence is not None and isinstance(sequence, np.ndarray) and sequence.ndim == 2 and sequence.shape[0] > 0:
            last = sequence[-1]
            obi = float(last[0])
            spread = float(last[1])
            lvl1 = float(last[2])
            ob_end = int(len(OB_KEYS))
            try:
                bid_slope = float(last[int(OB_KEYS.index("bid_slope"))])
            except Exception:
                pass
            try:
                offer_slope = float(last[int(OB_KEYS.index("offer_slope"))])
            except Exception:
                pass
            ret3 = float(last[ob_end + int(CD_KEYS.index("ret3"))])
            vol_accel = float(last[ob_end + int(CD_KEYS.index("vol_accel"))])
    except Exception:
        pass

    try:
        slope_signal = float(np.clip((float(bid_slope) - float(offer_slope)) * 5.0, -0.3, 0.3))
    except Exception:
        slope_signal = 0.0

    w_obi = float(w.get("w_obi", 0.55))
    w_lvl = float(w.get("w_lvl1", 0.30))
    w_sl = float(w.get("w_slope", 0.15))
    mom_scale = float(w.get("mom_scale", 50.0)) * float(max(0.0, mom_multiplier))
    mom_coef = float(w.get("mom_coef", 0.10))
    p_clip = float(w.get("pressure_clip", 0.48))
    sp_coef = float(w.get("spread_penalty_coef", 0.25))

    pressure_ob = (w_obi * obi) + (w_lvl * lvl1) + (w_sl * slope_signal)
    mom = max(-1.0, min(1.0, ret3 * mom_scale))
    vol_boost = max(0.0, min(1.0, (vol_accel - 1.0)))
    pressure = pressure_ob + (mom_coef * mom * (0.5 + 0.5 * vol_boost))

    try:
        spread_scale = float(confidence_spread_max_for_high or 1.0)
    except Exception:
        spread_scale = 1.0
    spread_scale = float(max(spread_scale, 1e-9))
    spread_penalty = max(0.0, min(0.25, (float(spread) / spread_scale) * sp_coef))
    raw = 0.5 + max(-p_clip, min(p_clip, pressure)) - spread_penalty
    prob = max(0.0, min(1.0, float(raw)))
    return float(prob), float(spread)


class TransformerPredictor:
    """TransformerPredictor.
"""
    def __init__(
        self,
        weights_path: Optional[str] = None,
        feature_dim: int = PAST_UNKNOWN_DIM,
        seq_len: int = 60,
        device: str = "cpu",
        buy_threshold: float = 0.62,
        sell_threshold: float = 0.38,
        confidence_high_margin: float = 0.15,
        confidence_mid_margin: float = 0.08,
        confidence_spread_max_for_high: float = 1.0,
        **kwargs,
    ):
        """Transformer 기반 수치 예측기를 초기화한다.

        Args:
            weights_path: 모델 가중치 파일 경로 (.pt). 없으면 rule-based fallback.
            feature_dim: 입력 피처 차원 수 (기본 PAST_UNKNOWN_DIM).
            seq_len: 시퀀스 길이 (기본 60).
            device: PyTorch 디바이스 문자열 ('cpu', 'cuda' 등).
            buy_threshold: BUY 신호 확률 임계값 (기본 0.62).
            sell_threshold: SELL 신호 확률 임계값 (기본 0.38).
            confidence_high_margin: HIGH 신뢰도 판정 마진 (기본 0.15).
            confidence_mid_margin: MEDIUM 신뢰도 판정 마진 (기본 0.08).
            confidence_spread_max_for_high: HIGH 신뢰도 허용 최대 스프레드 (기본 1.0).
            **kwargs: 모델 아키텍처 파라미터 (d_model, n_heads, n_layers, d_ff, dropout, pooling).
        """
        self._feature_dim = int(feature_dim)
        self._seq_len = int(seq_len)
        self._device = str(device)
        self._buy_threshold = float(buy_threshold)
        self._sell_threshold = float(sell_threshold)
        self._confidence_high_margin = float(confidence_high_margin)
        self._confidence_mid_margin = float(confidence_mid_margin)
        self._confidence_spread_max_for_high = float(confidence_spread_max_for_high)
        _kw = dict(kwargs or {})
        self._rule_based_weights = _merge_rule_based_weights(_kw.pop("rule_based_weights", None))
        try:
            self._rule_based_mom_multiplier = float(_kw.pop("rule_based_mom_multiplier", 1.0) or 1.0)
        except Exception:
            self._rule_based_mom_multiplier = 1.0
        kwargs = _kw
        self._model = None
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._warned_missing_norm = False
        self._warned_norm_dim_mismatch = False

        # PatchTST / Mamba 전용 키 포함 (PriceTransformer 는 무시)
        allowed_model_kw = {
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
            "dropout",
            "pooling",
            "patch_len",   # PatchTST 전용
            "stride",      # PatchTST 전용
            "d_state",     # Mamba 전용
        }
        model_init_kwargs: Dict[str, Any] = {}
        try:
            model_init_kwargs = {str(k): v for k, v in (kwargs or {}).items() if str(k) in allowed_model_kw}
        except Exception:
            model_init_kwargs = {}

        # model_class: 'transformer'(기본) 또는 'patch_tst'
        # config.json prediction.model_class 또는 kwargs["model_class"] 로 지정 가능
        self._model_class = str((kwargs or {}).get("model_class") or "transformer").strip().lower()

        path = str(weights_path or _DEFAULT_WEIGHTS)
        if Path(path).exists():
            try:
                # Backward-compat: older weights may have been trained with a different feature_dim.
                # Inspect state_dict first so we can skip loading (fallback) with a clear message.
                saved_feature_dim: Optional[int] = None
                try:
                    import torch

                    obj = torch.load(str(path), map_location=str(self._device), weights_only=False)
                    state = None
                    if isinstance(obj, dict) and "state_dict" in obj:
                        state = obj.get("state_dict")
                    else:
                        state = obj

                    if isinstance(obj, dict):
                        try:
                            xm = obj.get("x_mean")
                            xs = obj.get("x_std")
                            if xm is not None and xs is not None:
                                xm_arr = np.array(xm, dtype=np.float32).reshape(-1)
                                xs_arr = np.array(xs, dtype=np.float32).reshape(-1)
                                if int(len(xm_arr)) == int(self._feature_dim) and int(len(xs_arr)) == int(self._feature_dim):
                                    xs_arr = np.where(xs_arr > 1e-6, xs_arr, 1e-6).astype(np.float32)
                                    self._x_mean = xm_arr
                                    self._x_std = xs_arr
                        except Exception:
                            pass

                        try:
                            mk = obj.get("model_kwargs")
                            if isinstance(mk, dict) and mk:
                                # 체크포인트 아키텍처 설정 우선 복원
                                for k, v in mk.items():
                                    ks = str(k)
                                    if ks in allowed_model_kw:
                                        model_init_kwargs[ks] = v
                        except Exception:
                            pass

                    if isinstance(state, dict):
                        # PriceTransformer: input_proj.weight[1] = feature_dim
                        # PatchTSTModel:    patch_embed.proj.weight[1] = patch_len * feature_dim
                        w_input = state.get("input_proj.weight")
                        w_patch = state.get("patch_embed.proj.weight")
                        try:
                            if hasattr(w_input, "shape") and len(getattr(w_input, "shape")) == 2:
                                saved_feature_dim = int(w_input.shape[1])
                            elif hasattr(w_patch, "shape") and len(getattr(w_patch, "shape")) == 2:
                                # patch_embed.proj 입력 차원 = patch_len * feature_dim
                                pl = int(model_init_kwargs.get("patch_len") or 8)
                                saved_feature_dim = int(w_patch.shape[1]) // max(1, pl)
                                self._model_class = "patch_tst"
                        except Exception:
                            saved_feature_dim = None
                except Exception:
                    saved_feature_dim = None
                    state = None  # inner try 실패 시 state 미정의 방지

                if saved_feature_dim is not None and int(saved_feature_dim) != int(self._feature_dim):
                    logger.warning(
                        "[Predictor] Transformer weights feature_dim mismatch: saved=%s current=%s; skip loading and fallback to rule-based (%s)",
                        int(saved_feature_dim),
                        int(self._feature_dim),
                        path,
                    )
                    self._model = None
                    return

                # ── 모델 클래스 동적 선택 ────────────────────────────────────
                # state_dict 키로 자동 감지: patch_embed → patch_tst, blocks.0 → mamba
                if isinstance(state, dict):
                    if any(k.startswith("blocks.") for k in state):
                        self._model_class = "mamba"
                    elif any(k.startswith("patch_embed.") for k in state):
                        self._model_class = "patch_tst"

                if self._model_class == "patch_tst":
                    from .models.patch_tst_model import PatchTSTModel as _ModelCls
                    logger.info("[Predictor] model_class=patch_tst")
                elif self._model_class == "mamba":
                    from .models.mamba_model import MambaModel as _ModelCls  # type: ignore[assignment]
                    logger.info("[Predictor] model_class=mamba")
                else:
                    from .models.model import PriceTransformer as _ModelCls  # type: ignore[assignment]
                    logger.info("[Predictor] model_class=transformer")

                self._model = _ModelCls.load(
                    path,
                    feature_dim=int(self._feature_dim),
                    seq_len=int(self._seq_len),
                    device=str(self._device),
                    **model_init_kwargs,
                )
                logger.info("[Predictor] weights loaded: %s", path)
            except Exception as e:
                self._model = None
                logger.warning("[Predictor] Transformer load failed; fallback to rule-based: %s", e)
        else:
            logger.info("[Predictor] No weights (%s); using rule-based", path)

    def predict(
        self,
        *,
        input: ModelInput,
    ) -> TransformerPredictionResult:
        """Predict probability/signal from the latest features.

        Args:
            sequence: `(seq_len, feature_dim)` array from `features.build_sequence`.
            feature_snapshot: Latest orderbook feature dict.

        Returns:
            `TransformerPredictionResult`.
        """
        seq = getattr(input, "sequence", None)
        snap = dict(getattr(input, "feature_snapshot", None) or {})

        seq_norm = seq
        try:
            if isinstance(seq, np.ndarray) and seq.ndim == 2 and self._x_mean is not None and self._x_std is not None:
                if int(seq.shape[1]) == int(len(self._x_mean)):
                    seq_norm = (seq.astype(np.float32) - self._x_mean.reshape(1, -1)) / self._x_std.reshape(1, -1)
                else:
                    if not self._warned_norm_dim_mismatch:
                        self._warned_norm_dim_mismatch = True
                        logger.warning(
                            "[Predictor] normalization stats dim mismatch: seq_dim=%s stats_dim=%s; skip normalization",
                            int(seq.shape[1]),
                            int(len(self._x_mean)),
                        )
        except Exception:
            seq_norm = seq

        if self._x_mean is None or self._x_std is None:
            if (not self._warned_missing_norm) and self._model is not None:
                self._warned_missing_norm = True
                logger.warning(
                    "[Predictor] no normalization stats in checkpoint; "
                    "applying runtime z-score normalization as fallback. "
                    "Retrain with train_patch_tst.py to embed stats in checkpoint."
                )
            # [FIX-NORM-2] 체크포인트에 정규화 통계가 없는 구형 모델을 위한
            # 런타임 z-score 폴백. 배치 내 통계를 사용하므로 학습 시 통계와 다를 수
            # 있으나, 비정규화 원시값 입력으로 인한 Attention Saturation(prob=0.9993
            # 고착)을 방지하는 것이 목적이다.
            if isinstance(seq, np.ndarray) and seq.ndim == 2 and seq.shape[0] > 1:
                try:
                    run_mean = seq.mean(axis=0, keepdims=True).astype(np.float32)
                    run_std  = seq.std(axis=0,  keepdims=True).astype(np.float32)
                    run_std  = np.where(run_std > 1e-6, run_std, 1e-6).astype(np.float32)
                    seq_norm = (seq.astype(np.float32) - run_mean) / run_std
                except Exception:
                    seq_norm = seq

        # Guard: input feature dimension can vary by runtime configuration
        # (option_feature_set v1/v2, adaptive enabled/disabled). Do not assume
        # the global PAST_UNKNOWN_DIM is authoritative at inference time.
        try:
            if isinstance(seq, np.ndarray) and seq.ndim == 2:
                if int(seq.shape[1]) != int(self._feature_dim):
                    logger.warning(
                        "[Predictor] Transformer input feature_dim mismatch: got=%s expected=%s; fallback to rule-based",
                        int(seq.shape[1]),
                        int(self._feature_dim),
                    )
                    return self._rule_based(snap, seq)
        except Exception:
            pass

        if self._model is not None and seq is not None:
            try:
                import torch

                x_seq = seq_norm if (seq_norm is not None) else seq
                x = torch.tensor(x_seq[np.newaxis], dtype=torch.float32, device=str(self._device))
                with torch.no_grad():
                    prob_val = float(self._model(x).item())

                # [FIX-NORM-3] Sigmoid 포화값 감지 및 신뢰도 강등
                # prob > 0.998 또는 prob < 0.002 는 sigmoid(±6.2) 이상의 극단 logit을
                # 의미한다. 정규화 미적용이나 피처 스케일 문제로 발생하며, 이 상태의
                # 예측은 시장 정보를 반영하지 않으므로 confidence를 LOW로 강등한다.
                _SAT_HI = 0.998
                _SAT_LO = 1.0 - _SAT_HI  # 0.002
                _is_saturated = (prob_val >= _SAT_HI) or (prob_val <= _SAT_LO)

                spread = 0.0
                try:
                    if isinstance(seq, np.ndarray) and seq.ndim == 2 and seq.shape[0] > 0:
                        spread = float(seq[-1, 1])
                except Exception:
                    spread = float(snap.get("spread") or 0.0)

                signal, confidence = self._classify(prob_val, spread)

                if _is_saturated:
                    confidence = "LOW"
                    logger.warning(
                        "[Predictor] prob=%.4f is saturated (threshold=%.3f/%.3f) — "
                        "confidence forced to LOW. "
                        "Likely cause: normalization stats missing in checkpoint. "
                        "Retrain with train_patch_tst.py to resolve permanently.",
                        prob_val, _SAT_HI, _SAT_LO,
                    )
                return TransformerPredictionResult(
                    prob=round(float(prob_val), 4),
                    signal=str(signal),
                    confidence=str(confidence),
                    feature_snapshot=snap,
                )
            except Exception as e:
                logger.warning("[Predictor] inference failed; fallback to rule-based: %s", e)

        return self._rule_based(snap, seq)

    def _classify(self, prob: float, spread: float) -> tuple[str, str]:
        """BUY/SELL/HOLD 신호와 신뢰도 등급을 반환한다.

        Args:
            prob: 모델 예측 확률 (0.0~1.0). 0.5 초과 시 상승 방향.
            spread: bid-ask 스프레드 (포인트). HIGH 신뢰도 판정 시 사용.

        Returns:
            (signal, confidence) 튜플.
            signal: "BUY" | "SELL" | "HOLD"
            confidence: "HIGH" | "MEDIUM" | "LOW"
        """
        return _classify(
            prob=float(prob),
            spread=float(spread),
            buy_threshold=float(self._buy_threshold),
            sell_threshold=float(self._sell_threshold),
            confidence_high_margin=float(self._confidence_high_margin),
            confidence_mid_margin=float(self._confidence_mid_margin),
            confidence_spread_max_for_high=float(self._confidence_spread_max_for_high),
        )

    def _rule_based(
        self,
        snap: Dict[str, Any],
        sequence: Optional[np.ndarray],
    ) -> TransformerPredictionResult:
        """모델 가중치 없이 rule-based 예측 결과를 반환한다 (fallback 경로).

        Args:
            snap: 현재 시장 스냅샷 dict (피처 값 포함).
            sequence: 최근 분봉 시퀀스 numpy 배열 (없으면 None).

        Returns:
            TransformerPredictionResult (signal="HOLD", prob=0.5 기반 heuristic).
        """
        prob, spread = compute_rule_based_probability(
            snap,
            sequence,
            weights=self._rule_based_weights,
            mom_multiplier=float(getattr(self, "_rule_based_mom_multiplier", 1.0) or 1.0),
            confidence_spread_max_for_high=float(self._confidence_spread_max_for_high),
        )
        signal, confidence = self._classify(prob, spread)
        return TransformerPredictionResult(
            prob=round(float(prob), 4),
            signal=str(signal),
            confidence=str(confidence),
            feature_snapshot=snap,
        )


class NumericPredictor(Protocol):
    def predict(
        self,
        *,
        input: ModelInput,
    ) -> PredictionResult:
        ...


class RuleBasedPredictor:
    def __init__(
        self,
        *,
        buy_threshold: float = 0.62,
        sell_threshold: float = 0.38,
        confidence_high_margin: float = 0.15,
        confidence_mid_margin: float = 0.08,
        confidence_spread_max_for_high: float = 1.0,
        rule_based_weights: Optional[Mapping[str, float]] = None,
        rule_based_mom_multiplier: float = 1.0,
    ):
        self._buy_threshold = float(buy_threshold)
        self._sell_threshold = float(sell_threshold)
        self._confidence_high_margin = float(confidence_high_margin)
        self._confidence_mid_margin = float(confidence_mid_margin)
        self._confidence_spread_max_for_high = float(confidence_spread_max_for_high)
        self._rule_based_weights = _merge_rule_based_weights(rule_based_weights)
        try:
            self._rule_based_mom_multiplier = float(rule_based_mom_multiplier or 1.0)
        except Exception:
            self._rule_based_mom_multiplier = 1.0

    def _classify(self, prob: float, spread: float) -> tuple[str, str]:
        return _classify(
            prob=float(prob),
            spread=float(spread),
            buy_threshold=float(self._buy_threshold),
            sell_threshold=float(self._sell_threshold),
            confidence_high_margin=float(self._confidence_high_margin),
            confidence_mid_margin=float(self._confidence_mid_margin),
            confidence_spread_max_for_high=float(self._confidence_spread_max_for_high),
        )

    def predict(self, *, input: ModelInput) -> TransformerPredictionResult:
        snap = dict(getattr(input, "feature_snapshot", None) or {})
        seq = getattr(input, "sequence", None)
        prob, spread = compute_rule_based_probability(
            snap,
            seq,
            weights=self._rule_based_weights,
            mom_multiplier=self._rule_based_mom_multiplier,
            confidence_spread_max_for_high=self._confidence_spread_max_for_high,
        )
        signal, confidence = self._classify(prob, spread)
        return TransformerPredictionResult(
            prob=round(float(prob), 4),
            signal=str(signal),
            confidence=str(confidence),
            feature_snapshot=snap,
        )


class TFTPredictor:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        past_unknown_dim: int = PAST_UNKNOWN_DIM,
        future_known_dim: int = FUTURE_KNOWN_DIM,
        seq_len: int = 60,
        horizon: int = HORIZON_SEC,
        device: str = "cpu",
        buy_threshold: float = 0.62,
        sell_threshold: float = 0.38,
        **kwargs: Any,
    ):
        self._available = False
        self._buy_threshold = float(buy_threshold)
        self._sell_threshold = float(sell_threshold)
        self._device = str(device)
        self._past_unknown_dim = int(past_unknown_dim)
        self._future_known_dim = int(future_known_dim)
        self._model = None
        # confidence 파라미터 저장 — predict() 내 _classify 호출 시 getattr 폴백 없이 직접 참조
        # config.json의 confidence_high_margin 등이 TFT 신뢰도 판정에 정상 반영된다.
        self._confidence_high_margin: float = float(kwargs.get("confidence_high_margin", 0.15))
        self._confidence_mid_margin: float = float(kwargs.get("confidence_mid_margin", 0.08))
        self._confidence_spread_max_for_high: float = float(kwargs.get("confidence_spread_max_for_high", 1.0))

        path = str(weights_path or _DEFAULT_TFT_WEIGHTS)
        if Path(path).exists():
            try:
                # Backward-compat: older TFT weights may have different input dimensions.
                inferred_pu: Optional[int] = None
                inferred_fk: Optional[int] = None
                try:
                    import torch

                    state = torch.load(str(path), map_location=str(self._device), weights_only=True)
                    if isinstance(state, dict):
                        # Infer num_vars from the VSN weight_grn input layer:
                        # weight_grn.fc1: in_features = num_vars * d_model (+ static ctx), out_features = d_model
                        def _infer_vars(prefix: str) -> Optional[int]:
                            w = state.get(f"{prefix}.weight_grn.fc1.weight")
                            if w is None or not hasattr(w, "shape") or len(getattr(w, "shape")) != 2:
                                return None
                            try:
                                out_dim = int(w.shape[0])
                                in_dim = int(w.shape[1])
                                if out_dim <= 0 or in_dim <= 0:
                                    return None
                                if in_dim % out_dim != 0:
                                    return None
                                return int(in_dim // out_dim)
                            except Exception:
                                return None

                        inferred_pu = _infer_vars("vsn_past_unknown")
                        inferred_fk = _infer_vars("vsn_past_known") or _infer_vars("vsn_future_known")
                except Exception:
                    inferred_pu = None
                    inferred_fk = None

                if inferred_pu is not None and int(inferred_pu) != int(past_unknown_dim):
                    logger.warning(
                        "[TFTPredictor] TFT weights past_unknown_dim mismatch: saved=%s current=%s; skip loading and disable TFT (%s)",
                        int(inferred_pu),
                        int(past_unknown_dim),
                        path,
                    )
                    self._model = None
                    self._available = False
                    return

                if inferred_fk is not None and int(inferred_fk) != int(future_known_dim):
                    logger.warning(
                        "[TFTPredictor] TFT weights future_known_dim mismatch: saved=%s current=%s; skip loading and disable TFT (%s)",
                        int(inferred_fk),
                        int(future_known_dim),
                        path,
                    )
                    self._model = None
                    self._available = False
                    return

                from .models.tft_model import TemporalFusionTransformer

                self._model = TemporalFusionTransformer.load(
                    path,
                    past_unknown_dim=int(past_unknown_dim),
                    future_known_dim=int(future_known_dim),
                    seq_len=int(seq_len),
                    horizon=int(horizon),
                    device=str(device),
                    **kwargs,
                )
                self._available = True
                logger.info("[TFTPredictor] TFT weights loaded: %s", path)
            except Exception as e:
                self._model = None
                self._available = False
                logger.warning("[TFTPredictor] load failed: %s", e)
        else:
            logger.info("[TFTPredictor] No weights (%s); disabled", path)

    @property
    def is_available(self) -> bool:
        return bool(self._available and self._model is not None)

    def predict(self, *, input: ModelInput) -> PredictionResult:
        seq = getattr(input, "sequence", None)
        pk = getattr(input, "past_known", None)
        fk = getattr(input, "future_known", None)
        snap = dict(getattr(input, "feature_snapshot", None) or {})

        if not self.is_available:
            return TransformerPredictionResult(
                prob=0.5,
                signal="HOLD",
                confidence="LOW",
                feature_snapshot=snap,
            )

        if seq is None or pk is None or fk is None:
            return TransformerPredictionResult(
                prob=0.5,
                signal="HOLD",
                confidence="LOW",
                feature_snapshot=snap,
            )

        # Guard: do not rely on PAST_UNKNOWN_DIM/FUTURE_KNOWN_DIM as runtime truth.
        # If arrays do not match the configured dims, degrade gracefully.
        try:
            if isinstance(seq, np.ndarray) and seq.ndim == 2:
                if int(seq.shape[1]) != int(self._past_unknown_dim):
                    logger.warning(
                        "[TFTPredictor] past_unknown input dim mismatch: got=%s expected=%s; returning HOLD",
                        int(seq.shape[1]),
                        int(self._past_unknown_dim),
                    )
                    return TransformerPredictionResult(
                        prob=0.5,
                        signal="HOLD",
                        confidence="LOW",
                        feature_snapshot=snap,
                    )
            if isinstance(pk, np.ndarray) and pk.ndim == 2:
                if int(pk.shape[1]) != int(self._future_known_dim):
                    logger.warning(
                        "[TFTPredictor] past_known input dim mismatch: got=%s expected=%s; returning HOLD",
                        int(pk.shape[1]),
                        int(self._future_known_dim),
                    )
                    return TransformerPredictionResult(
                        prob=0.5,
                        signal="HOLD",
                        confidence="LOW",
                        feature_snapshot=snap,
                    )
            if isinstance(fk, np.ndarray) and fk.ndim == 2:
                if int(fk.shape[1]) != int(self._future_known_dim):
                    logger.warning(
                        "[TFTPredictor] future_known input dim mismatch: got=%s expected=%s; returning HOLD",
                        int(fk.shape[1]),
                        int(self._future_known_dim),
                    )
                    return TransformerPredictionResult(
                        prob=0.5,
                        signal="HOLD",
                        confidence="LOW",
                        feature_snapshot=snap,
                    )
        except Exception:
            pass

        try:
            import torch

            pu_t = torch.tensor(seq[np.newaxis], dtype=torch.float32, device=str(self._device))
            pk_t = torch.tensor(pk[np.newaxis], dtype=torch.float32, device=str(self._device))
            fk_t = torch.tensor(fk[np.newaxis], dtype=torch.float32, device=str(self._device))

            with torch.no_grad():
                prob_val = float(self._model(pu_t, pk_t, fk_t).item())

            spread = 0.0
            try:
                if isinstance(seq, np.ndarray) and seq.ndim == 2 and seq.shape[0] > 0:
                    spread = float(seq[-1, 1])
            except Exception:
                spread = float(snap.get("spread") or 0.0)

            signal, confidence = _classify(
                prob=float(prob_val),
                spread=float(spread),
                buy_threshold=float(self._buy_threshold),
                sell_threshold=float(self._sell_threshold),
                confidence_high_margin=float(self._confidence_high_margin),
                confidence_mid_margin=float(self._confidence_mid_margin),
                confidence_spread_max_for_high=float(self._confidence_spread_max_for_high),
            )
            return TransformerPredictionResult(
                prob=round(float(prob_val), 4),
                signal=str(signal),
                confidence=str(confidence),
                feature_snapshot=snap,
            )
        except Exception as e:
            logger.warning("[TFTPredictor] inference failed: %s", e)
            return TransformerPredictionResult(
                prob=0.5,
                signal="HOLD",
                confidence="LOW",
                feature_snapshot=snap,
            )


@dataclass
class EnsemblePredictionResult:
    prob: float
    signal: str
    confidence: str
    transformer_prob: float
    tft_prob: Optional[float]
    ensemble_method: str
    agreement: bool
    feature_snapshot: Dict[str, Any]
    prob_lower: Optional[float] = None   # Conformal Prediction 하한 [0,1]
    prob_upper: Optional[float] = None   # Conformal Prediction 상한 [0,1]


class EnsemblePredictor:
    def __init__(
        self,
        transformer_predictor: TransformerPredictor,
        tft_predictor: TFTPredictor,
        transformer_weight: float = 0.5,
        buy_threshold: float = 0.62,
        sell_threshold: float = 0.38,
        confidence_high_margin: float = 0.15,
        confidence_mid_margin: float = 0.08,
        confidence_spread_max_for_high: float = 1.0,
        disagreement_hold: bool = True,
        disagreement_hold_prob_diff_max: float = 0.1,
        disagreement_hold_prob_diff_max_by_regime: Optional[Mapping[str, float]] = None,
        ensemble_agreement_confidence_boost: bool = True,
        ensemble_agreement_prob_diff_max: float = 0.06,
        conformal_alpha: float = 0.1,
        conformal_path: Optional[str] = None,
        confidence_conformal_width_max_for_high: float = 0.35,
        confidence_conformal_width_max_for_medium: float = 0.55,
        mamba_predictor: Optional["TransformerPredictor"] = None,
        mamba_weight: float = 0.33,
    ):
        self._transformer = transformer_predictor
        self._tft = tft_predictor
        self._mamba = mamba_predictor          # None이면 Mamba 비활성
        self._mamba_weight = float(max(0.0, min(1.0, mamba_weight)))
        self._w_transformer = float(transformer_weight)
        self._adaptive_weight_tracker = AdaptiveEnsembleWeightTracker(window=20)
        self._buy_threshold = float(buy_threshold)
        self._sell_threshold = float(sell_threshold)
        self._confidence_high_margin = float(confidence_high_margin)
        self._confidence_mid_margin = float(confidence_mid_margin)
        self._confidence_spread_max_for_high = float(confidence_spread_max_for_high)
        self._disagreement_hold = bool(disagreement_hold)
        self._disagreement_hold_prob_diff_max = float(disagreement_hold_prob_diff_max)
        self._disagreement_hold_prob_diff_max_by_regime: Optional[Dict[str, float]] = None
        if isinstance(disagreement_hold_prob_diff_max_by_regime, Mapping):
            try:
                self._disagreement_hold_prob_diff_max_by_regime = {
                    str(k).strip().upper(): float(v)
                    for k, v in disagreement_hold_prob_diff_max_by_regime.items()
                }
            except Exception:
                self._disagreement_hold_prob_diff_max_by_regime = None
        self._ensemble_agreement_confidence_boost = bool(ensemble_agreement_confidence_boost)
        self._ensemble_agreement_prob_diff_max = float(ensemble_agreement_prob_diff_max)
        self._current_regime: Optional[str] = None
        self._confidence_conformal_width_max_for_high = float(confidence_conformal_width_max_for_high)
        self._confidence_conformal_width_max_for_medium = float(confidence_conformal_width_max_for_medium)
        self._warned_tft_unavailable = False
        self._warned_transformer_rule_based = False
        self._warned_mamba_unavailable = False

        # ── Conformal Prediction 초기화 ────────────────────────────────────
        self._conformal = None
        try:
            from .conformal import ConformalPredictor
            path = str(conformal_path or "")
            self._conformal = ConformalPredictor.load_or_create(
                path if path else "__none__",
                alpha=float(conformal_alpha),
            )
        except Exception as _ce:
            logger.debug("[EnsemblePredictor] conformal init 실패 (구간 비활성): %s", _ce)

    # [IMP-ENS-02] 레짐별 초기 가중치 편향 매핑.
    # 피드백 데이터 부족 초기에는 레짐에 맞는 모델을 우선 신뢰한다.
    # STRONG_TREND → Transformer(추세 추종) 우선, RANGE → TFT(단기 평균회귀) 우선.
    _REGIME_BIAS: dict = {
        "STRONG_UP":   0.65,
        "STRONG_DOWN": 0.65,
        "WEAK_UP":     0.55,
        "WEAK_DOWN":   0.55,
        "RANGE":       0.40,  # TFT 우선
    }

    def _effective_disagreement_hold_max(self) -> float:
        """레짐별 오버라이드가 있으면 사용, 없으면 기본 ``_disagreement_hold_prob_diff_max``."""
        base = float(self._disagreement_hold_prob_diff_max)
        m = getattr(self, "_disagreement_hold_prob_diff_max_by_regime", None)
        if not isinstance(m, dict) or not m:
            return base
        r = getattr(self, "_current_regime", None)
        rk = str(r or "").strip().upper()
        if rk and rk in m:
            try:
                return float(m[rk])
            except Exception:
                return base
        return base

    def set_regime(self, regime: Optional[str]) -> None:
        """[IMP-ENS-02] 현재 시장 레짐을 설정한다.

        - AdaptiveEnsembleWeightTracker의 창 크기를 레짐에 맞게 조정한다.
        - 피드백 이력이 없으면 레짐 초기 편향으로 _w_transformer를 설정한다.
        - dte_weight_norm > 0.5(만기주)이면 window=8로 단축한다.

        Args:
            regime: "STRONG_UP" / "STRONG_DOWN" / "WEAK_UP" / "WEAK_DOWN" / "RANGE" / None
        """
        try:
            r = str(regime or "").strip().upper()
            self._current_regime = r if r else None
            # 레짐별 창 크기
            if r in ("STRONG_UP", "STRONG_DOWN"):
                new_window = 12
            elif r == "RANGE":
                new_window = 30
            else:
                new_window = 20
            try:
                self._adaptive_weight_tracker.set_window(new_window)
            except Exception:
                pass

            # 피드백 이력이 아직 없으면 레짐 편향을 초기 가중치로 설정
            try:
                tw, fw = self._adaptive_weight_tracker.get_weights()
                # get_weights()가 (0.5, 0.5)를 반환하면 이력 없음으로 간주
                is_uninformed = abs(tw - 0.5) < 1e-6 and abs(fw - 0.5) < 1e-6
            except Exception:
                is_uninformed = True

            if is_uninformed and r in self._REGIME_BIAS:
                self._w_transformer = float(self._REGIME_BIAS[r])
        except Exception:
            pass

    def set_dte_window(self, dte_weight_norm: float) -> None:
        """[IMP-ENS-02] 만기 근접도에 따라 창 크기를 단축한다.

        dte_weight_norm > 0.5(만기 약 2일 이내)이면 window=8로 설정한다.
        """
        try:
            if float(dte_weight_norm or 0.0) > 0.5:
                self._adaptive_weight_tracker.set_window(8)
        except Exception:
            pass

    def update_adaptive_weights(
        self,
        *,
        transformer_correct: bool,
        tft_correct: bool,
        transformer_weight: float = 1.0,
        tft_weight: float = 1.0,
        transformer_prob: Optional[float] = None,
        tft_prob: Optional[float] = None,
    ) -> None:
        try:
            # [IMP-ENS-01] transformer_prob/tft_prob를 함께 전달해 Brier Score 점수화
            self._adaptive_weight_tracker.update(
                transformer_correct=bool(transformer_correct),
                tft_correct=bool(tft_correct),
                transformer_weight=float(transformer_weight),
                tft_weight=float(tft_weight),
                transformer_prob=transformer_prob,
                tft_prob=tft_prob,
            )
            w_t, _w_f = self._adaptive_weight_tracker.get_weights()
            self._w_transformer = max(0.0, min(1.0, float(w_t)))
        except Exception:
            pass

    def get_transformer_weight(self) -> float:
        try:
            return float(self._w_transformer)
        except Exception:
            return 0.5

    def reset_adaptive_weights(self) -> None:
        try:
            tr = getattr(self, "_adaptive_weight_tracker", None)
            if tr is not None and callable(getattr(tr, "reset", None)):
                tr.reset()
        except Exception:
            pass
        try:
            self._w_transformer = 0.5
        except Exception:
            pass

    def predict(self, *, input: ModelInput) -> EnsemblePredictionResult:
        snap = dict(getattr(input, "feature_snapshot", None) or {})

        t_res = self._transformer.predict(input=input)

        transformer_has_torch_model = bool(getattr(self._transformer, "_model", None) is not None)
        if (not transformer_has_torch_model) and (not self._warned_transformer_rule_based):
            self._warned_transformer_rule_based = True
            logger.warning(
                "[EnsemblePredictor] Transformer is in rule-based mode (no/invalid weights or torch unavailable); ensemble quality may degrade"
            )

        # ── Mamba 예측 (활성화된 경우) ────────────────────────────────────
        mamba_prob: Optional[float] = None
        mamba_available = False
        if self._mamba is not None:
            try:
                mamba_has_model = bool(getattr(self._mamba, "_model", None) is not None)
                if mamba_has_model:
                    m_res = self._mamba.predict(input=input)
                    mamba_prob = float(m_res.prob)
                    mamba_available = True
                else:
                    if not self._warned_mamba_unavailable:
                        self._warned_mamba_unavailable = True
                        logger.warning("[EnsemblePredictor] Mamba 가중치 없음 — 앙상블에서 제외")
            except Exception as _me:
                if not self._warned_mamba_unavailable:
                    self._warned_mamba_unavailable = True
                    logger.warning("[EnsemblePredictor] Mamba 추론 실패: %s", _me)

        # ── TFT + Mamba 조합으로 앙상블 가중치 계산 ──────────────────────
        if self._tft.is_available and mamba_available:
            # 3자 앙상블: Transformer + TFT + Mamba
            f_res = self._tft.predict(input=input)
            tft_prob: Optional[float] = float(f_res.prob)
            # 가중치 정규화: w_t + w_f + w_m = 1.0
            w_m = float(self._mamba_weight)
            w_t_raw = max(0.0, min(1.0, float(self._w_transformer)))
            # Transformer 와 TFT 가 남은 비중을 공유
            w_tf_total = max(1e-9, 1.0 - w_m)
            w_t = w_t_raw * w_tf_total
            w_f = (1.0 - w_t_raw) * w_tf_total
            ens_prob = w_t * float(t_res.prob) + w_f * float(tft_prob) + w_m * float(mamba_prob)
            method = "3way_ensemble"
        elif self._tft.is_available:
            # 기존 2자 앙상블: Transformer + TFT
            f_res = self._tft.predict(input=input)
            tft_prob = float(f_res.prob)
            w_t = max(0.0, min(1.0, float(self._w_transformer)))
            w_f = 1.0 - w_t
            ens_prob = (w_t * float(t_res.prob)) + (w_f * float(tft_prob))
            method = "weighted_avg"
        elif mamba_available:
            # 2자 앙상블: Transformer + Mamba
            tft_prob = None
            w_m = float(self._mamba_weight)
            w_t = 1.0 - w_m
            ens_prob = w_t * float(t_res.prob) + w_m * float(mamba_prob)
            method = "transformer_mamba"
        else:
            # Transformer 단독
            if not self._warned_tft_unavailable:
                self._warned_tft_unavailable = True
                logger.warning(
                    "[EnsemblePredictor] TFT is unavailable (no/invalid weights, torch missing, or dim mismatch); degrading to transformer-only"
                )
            tft_prob = None
            ens_prob = float(t_res.prob)
            method = "transformer_only" if transformer_has_torch_model else "rule_based_only"

        spread = float(snap.get("spread") or 0.0)
        try:
            if input.sequence is not None and isinstance(input.sequence, np.ndarray) and input.sequence.ndim == 2:
                spread = float(input.sequence[-1, 1])
        except Exception:
            pass

        # ── Conformal 구간 (신뢰도 보정에 먼저 사용) ─────────────────────────
        prob_lower: Optional[float] = None
        prob_upper: Optional[float] = None
        try:
            if self._conformal is not None and self._conformal.is_calibrated():
                prob_lower, prob_upper = self._conformal.predict_interval(float(ens_prob))
                prob_lower = round(float(prob_lower), 4)
                prob_upper = round(float(prob_upper), 4)
        except Exception as _ce:
            logger.debug("[EnsemblePredictor] conformal 구간 계산 실패: %s", _ce)

        signal, confidence = _classify(
            prob=float(ens_prob),
            spread=float(spread),
            buy_threshold=float(self._buy_threshold),
            sell_threshold=float(self._sell_threshold),
            confidence_high_margin=float(self._confidence_high_margin),
            confidence_mid_margin=float(self._confidence_mid_margin),
            confidence_spread_max_for_high=float(self._confidence_spread_max_for_high),
        )
        confidence = adjust_confidence_by_conformal_interval_width(
            confidence,
            prob_lower,
            prob_upper,
            width_max_for_high=float(self._confidence_conformal_width_max_for_high),
            width_max_for_medium=float(self._confidence_conformal_width_max_for_medium),
        )

        agreement = False
        if tft_prob is not None:
            t_dir = "BUY" if float(t_res.prob) >= float(self._buy_threshold) else (
                "SELL" if float(t_res.prob) <= float(self._sell_threshold) else "HOLD"
            )
            f_dir = "BUY" if float(tft_prob) >= float(self._buy_threshold) else (
                "SELL" if float(tft_prob) <= float(self._sell_threshold) else "HOLD"
            )
            agreement = (t_dir == f_dir)

        # 연속 확률 거리: 참여 모델 간 최대 쌍별 |Δp| (방향 일치와 무관)
        component_probs: List[float] = [float(t_res.prob)]
        try:
            if tft_prob is not None:
                component_probs.append(float(tft_prob))
            if mamba_available and mamba_prob is not None:
                component_probs.append(float(mamba_prob))
        except Exception:
            component_probs = [float(t_res.prob)]
        prob_diff = _max_pairwise_abs_diff(component_probs)
        disagreement_fired = False
        _thr = float(self._effective_disagreement_hold_max())
        if self._disagreement_hold and len(component_probs) >= 2:
            if prob_diff >= _thr:
                signal = "HOLD"
                confidence = "LOW"
                method = "disagreement_hold"
                disagreement_fired = True

        if (
            not disagreement_fired
            and self._ensemble_agreement_confidence_boost
            and tft_prob is not None
            and agreement
            and prob_diff < float(self._ensemble_agreement_prob_diff_max)
            and confidence == "MEDIUM"
            and float(spread or 0.0) <= float(self._confidence_spread_max_for_high)
        ):
            confidence = "HIGH"
            method = f"{method}_agreement_boost"

        return EnsemblePredictionResult(
            prob=round(float(ens_prob), 4),
            signal=str(signal),
            confidence=str(confidence),
            transformer_prob=round(float(t_res.prob), 4),
            tft_prob=round(float(tft_prob), 4) if tft_prob is not None else None,
            ensemble_method=str(method),
            agreement=bool(agreement),
            feature_snapshot=snap,
            prob_lower=prob_lower,
            prob_upper=prob_upper,
        )


def create_numeric_predictor(
    *,
    numeric_predictor: str = "transformer",
    buy_threshold: float = 0.62,
    sell_threshold: float = 0.38,
    transformer_weights_path: Optional[str] = None,
    tft_weights_path: Optional[str] = None,
    tft_horizon: int = 300,
    transformer_weight: float = 0.5,
    disagreement_hold: bool = True,
    disagreement_hold_prob_diff_max: float = 0.1,
    disagreement_hold_prob_diff_max_by_regime: Optional[Mapping[str, float]] = None,
    ensemble_agreement_confidence_boost: bool = True,
    ensemble_agreement_prob_diff_max: float = 0.06,
    confidence_conformal_width_max_for_high: float = 0.35,
    confidence_conformal_width_max_for_medium: float = 0.55,
    conformal_alpha: float = 0.1,
    conformal_path: Optional[str] = None,
    mamba_enabled: bool = False,
    mamba_weights_path: Optional[str] = None,
    mamba_weight: float = 0.33,
    **kwargs: Any,
) -> NumericPredictor:
    mode = str(numeric_predictor or "transformer").strip().lower()
    if mode == "combined":
        mode = "ensemble"
    if mode == "rule_based":
        return RuleBasedPredictor(
            buy_threshold=float(buy_threshold),
            sell_threshold=float(sell_threshold),
            confidence_high_margin=float(kwargs.get("confidence_high_margin", 0.15)),
            confidence_mid_margin=float(kwargs.get("confidence_mid_margin", 0.08)),
            confidence_spread_max_for_high=float(kwargs.get("confidence_spread_max_for_high", 1.0)),
            rule_based_weights=kwargs.get("rule_based_weights"),
            rule_based_mom_multiplier=float(kwargs.get("rule_based_mom_multiplier", 1.0) or 1.0),
        )
    if mode == "tft":
        return TFTPredictor(
            weights_path=str(tft_weights_path) if tft_weights_path else None,
            horizon=int(tft_horizon),
            buy_threshold=float(buy_threshold),
            sell_threshold=float(sell_threshold),
            **kwargs,
        )
    if mode == "ensemble":
        t_pred = TransformerPredictor(
            weights_path=str(transformer_weights_path) if transformer_weights_path else None,
            buy_threshold=float(buy_threshold),
            sell_threshold=float(sell_threshold),
            **kwargs,
        )
        f_pred = TFTPredictor(
            weights_path=str(tft_weights_path) if tft_weights_path else None,
            horizon=int(tft_horizon),
            buy_threshold=float(buy_threshold),
            sell_threshold=float(sell_threshold),
            **kwargs,
        )
        # ── Mamba ON/OFF ──────────────────────────────────────────────────
        # mamba_enabled=True 이고 mamba_weights_path 가 있을 때만 생성.
        # 가중치 없이 활성화하면 TransformerPredictor(rule-based)로 생성되어
        # 앙상블 품질이 저하되므로 경로 유무를 이중 확인한다.
        m_pred: Optional[TransformerPredictor] = None
        _mamba_path = str(mamba_weights_path or "").strip()
        if bool(mamba_enabled) and _mamba_path:
            try:
                m_pred = TransformerPredictor(
                    weights_path=_mamba_path,
                    buy_threshold=float(buy_threshold),
                    sell_threshold=float(sell_threshold),
                    model_class="mamba",
                    **{k: v for k, v in kwargs.items()
                       if k not in ("model_class", "patch_len", "stride")},
                )
                logger.info("[create_numeric_predictor] Mamba 앙상블 참여: %s", _mamba_path)
            except Exception as _me:
                logger.warning("[create_numeric_predictor] Mamba 초기화 실패 (제외): %s", _me)
                m_pred = None
        elif bool(mamba_enabled) and not _mamba_path:
            logger.warning(
                "[create_numeric_predictor] mamba_enabled=true 이지만 mamba_weights_path 가 비어 있어 Mamba를 비활성화합니다. "
                "train_mamba.py 로 훈련 후 경로를 설정하세요."
            )

        return EnsemblePredictor(
            transformer_predictor=t_pred,
            tft_predictor=f_pred,
            transformer_weight=float(transformer_weight),
            buy_threshold=float(buy_threshold),
            sell_threshold=float(sell_threshold),
            confidence_high_margin=float(kwargs.get("confidence_high_margin", 0.15)),
            confidence_mid_margin=float(kwargs.get("confidence_mid_margin", 0.08)),
            confidence_spread_max_for_high=float(kwargs.get("confidence_spread_max_for_high", 1.0)),
            disagreement_hold=bool(disagreement_hold),
            disagreement_hold_prob_diff_max=float(disagreement_hold_prob_diff_max),
            disagreement_hold_prob_diff_max_by_regime=disagreement_hold_prob_diff_max_by_regime,
            ensemble_agreement_confidence_boost=bool(ensemble_agreement_confidence_boost),
            ensemble_agreement_prob_diff_max=float(ensemble_agreement_prob_diff_max),
            confidence_conformal_width_max_for_high=float(confidence_conformal_width_max_for_high),
            confidence_conformal_width_max_for_medium=float(confidence_conformal_width_max_for_medium),
            conformal_alpha=float(conformal_alpha),
            conformal_path=str(conformal_path) if conformal_path else None,
            mamba_predictor=m_pred,
            mamba_weight=float(mamba_weight),
        )
    return TransformerPredictor(
        weights_path=str(transformer_weights_path) if transformer_weights_path else None,
        buy_threshold=float(buy_threshold),
        sell_threshold=float(sell_threshold),
        **kwargs,
    )
