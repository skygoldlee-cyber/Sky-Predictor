"""차트 엔진 모듈

OHLC + ZigZag 피봇 계산을 담당하는 순수 파이썬 클래스.
"""

from __future__ import annotations

import logging
import hashlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    pass

import numpy as np
import pandas as pd
from gui.utils.pivot_probability import HistoricalPivot

logger = logging.getLogger(__name__)


class ChartEngine:
    """OHLC + ZigZag 피봇 계산을 담당하는 순수 파이썬 클래스."""

    def __init__(self) -> None:
        self._zz: Optional[Any]     = None
        self._zz_cfg: Optional[Any] = None
        # [SSOT] set_zigzag()로 외부 주입된 경우 True → _init_zigzag() 재생성 금지
        self._zz_external: bool = False
        self._last_sig: Optional[tuple] = None   # 재계산 방지 캐시 키
        self._trade_markers_enabled = False  # 거래 마커 활성화 플래그
        self._trade_events: List[Dict[str, Any]] = []  # 거래 이벤트 리스트
        self._current_data_source: Optional[str] = None  # 현재 데이터 소스
        self._pivot_candidate_callback: Optional[callable] = None  # 피봇 후보 콜백
        self._max_bars: int = 500  # 렌더 윈도우 상한 (동적 조정)
        self._minutes: int = 0  # 현재 분봉 범위 (피봇 마커 재렌더용)
        self._last_completed_ts: Optional[pd.Timestamp] = None  # 마지막 완결봉 타임스탬프 (연속성 판단용)
        self._anchor_ts: Optional[pd.Timestamp] = None  # max_bars 슬라이딩 앵커 타임스탬프
        # ── [FIX-5] ZigZag 상태 캐시 ───────────────────────────────────────────────
        self._zz_state_cache: Dict[str, Any] = {}  # ZigZag 상태 캐시 (서명 기반)
        self._replay_signature: Optional[str] = None  # replay 기준 서명
        self._confirmed_pivots_cache: List[Dict[str, Any]] = []  # 확정 피봇 캐시
        # ── 레짐 기반 파라미터 조정 ────────────────────────────────────────────────
        self._regime_mapper: Optional[Any] = None  # RegimeParamMapper 인스턴스
        self._adaptive_enabled: bool = False  # Adaptive 모드 활성화 상태
        # ── SuperTrend ────────────────────────────────────────────────────────────
        self._st: Optional[Any] = None  # AdaptiveSuperTrend 인스턴스
        self._st_cfg: Optional[Any] = None  # AdaptiveSuperTrendConfig
        self._st_external: bool = False          # [SSOT] 외부 주입 시 True → _init_supertrend 재생성 금지
        self._st_external_origin: Optional[Any] = None  # [SSOT] 동일성 체크용 원본 참조
        self._st_cache_sig: Optional[tuple] = None  # SuperTrend 캐시 서명
        self._st_cache_values: Optional[list] = None  # 완결봉 SuperTrend 값 캐시
        self._st_cache_dirs: Optional[list] = None    # 완결봉 SuperTrend 방향 캐시
        self._st_fed_bars: int = 0   # BUG1 수정: 마지막으로 _st에 feed한 완결봉 수
        # ────────────────────────────────────────────────────────────────────────────
    
    def set_zigzag(self, zz: Any, data_source: Optional[str] = None) -> None:
        """외부 ZigZag 인스턴스 설정.

        Args:
            zz: AdaptiveZigZag 인스턴스
            data_source: 이 ZigZag가 어느 소스(kospi/futures)용인지.
                         지정 시 _current_data_source를 함께 업데이트하여
                         compute() 내 데이터소스 변경 블록이 재발동하지 않게 방지.
        """
        self._zz = zz
        self._zz_external_origin = zz  # [FIX] 원본 보존 — _prepare_refresh 동일성 체크용
        self._zz_external = True  # [SSOT] 외부 주입 — _init_zigzag() 재생성 금지
        self._last_sig = None  # 캐시 강제 무효화
        self._last_completed_ts = None  # 완결봉 타임스탬프 초기화
        self._anchor_ts = None  # 앵커 타임스탬프 초기화
        self._zz_state_cache = {}  # ZigZag 상태 캐시 초기화
        self._replay_signature = None  # replay 서명 초기화
        self._confirmed_pivots_cache = []  # 확정 피봇 캐시 초기화
        self._st_cache_sig = None  # SuperTrend 캐시 서명 초기화
        self._st_cache_values = None  # SuperTrend 값 캐시 초기화
        self._st_cache_dirs = None  # SuperTrend 방향 캐시 초기화
        self._st_fed_bars = 0  # BUG1 수정: feed한 완결봉 수 초기화
        # [BUG-FIX] data_source 지정 시 _current_data_source 동기화
        # 미지정 시 compute()의 소스 변경 블록이 _zz=None으로 덮어써
        # set_zigzag 효과가 사라지는 것을 방지
        if data_source is not None:
            self._current_data_source = data_source

    def set_supertrend(self, st: Any, data_source: Optional[str] = None) -> None:
        """[SSOT] 외부(AdaptiveIndicatorManager)에서 계산된 AdaptiveSuperTrend 인스턴스 주입.

        주입 후 compute()는 self._st 인스턴스에서 df 컬럼만 추출하고
        자체 replay 계산을 건너뛴다.

        set_zigzag()와 동일한 패턴:
          - _st_external=True 설정으로 _init_supertrend() 재생성 금지
          - _st_external_origin 보존으로 _prepare_refresh() 동일성 체크
          - 모든 캐시 무효화
        """
        self._st = st
        self._st_external = True
        self._st_external_origin = st  # 동일성 체크용 원본 보존
        # 캐시 전체 무효화 (이전 자체 계산 결과 폐기)
        self._st_cache_sig    = None
        self._st_cache_values = None
        self._st_cache_dirs   = None
        self._st_fed_bars     = 0
        if data_source is not None:
            self._current_data_source = data_source
        logger.info(
            "[ChartEngine] SuperTrend 외부 주입 (SSOT): ds=%s id=%s",
            data_source, id(st),
        )

    def set_pivot_candidate_callback(self, callback: callable) -> None:
        """피봇 후보 콜백 설정."""
        self._pivot_candidate_callback = callback
    
    def set_max_bars(self, minutes: int) -> None:
        """렌더 윈도우 상한 동적 조정.

        Args:
            minutes: 표시할 분봉 수 (9999는 장전체)
        """
        self._minutes = minutes  # 분봉 범위 저장 (피봇 마커 재렌더용)
        self._anchor_ts = None  # 이전 범위 앵커 무효화
        self._last_sig = None   # compute() 캐시 무효화
        if minutes >= 9999:
            self._max_bars = 1000
        elif minutes >= 120:
            self._max_bars = minutes + 100
        else:
            self._max_bars = 500

    # ── ZigZag 초기화 ────────────────────────────────────────────────────────

    def _init_supertrend(self, cfg: Optional[Any] = None) -> None:
        """indicators.AdaptiveSuperTrend 인스턴스를 생성한다.

        [SSOT] set_supertrend()로 외부 주입된 인스턴스가 있으면 재생성하지 않는다.
        _zz_external 가드와 동일한 패턴.
        """
        # [SSOT] 외부 주입 인스턴스 보존 — _init_supertrend 재생성 금지
        if self._st_external and self._st is not None:
            logger.debug(
                "[ChartEngine] _init_supertrend: 외부 주입 ST 보존 (id=%s)", id(self._st)
            )
            return

        AdaptiveSuperTrend = None
        AdaptiveSuperTrendConfig = None

        try:
            from indicators import AdaptiveSuperTrend, AdaptiveSuperTrendConfig
        except Exception as e:
            logger.warning("[ChartEngine] AdaptiveSuperTrend import failed: %s — 슈퍼트렌드 비활성", e)

        if AdaptiveSuperTrend is None:
            self._st = None
            self._st_cfg = None
            return

        # config에서 슈퍼트렌드 설정 가져오기
        st_cfg_dict = {}
        if cfg is not None:
            try:
                adaptive_indicator = getattr(cfg, "adaptive_indicator", None)
                if adaptive_indicator is not None:
                    supertrend_cfg = getattr(adaptive_indicator, "supertrend", None)
                    if supertrend_cfg is not None:
                        st_cfg_dict = supertrend_cfg.__dict__ if hasattr(supertrend_cfg, "__dict__") else supertrend_cfg
            except Exception as e:
                logger.debug("[ChartEngine] 슈퍼트렌드 설정 가져오기 실패: %s", e)

        try:
            self._st_cfg = AdaptiveSuperTrendConfig(
                atr_min_period=int(st_cfg_dict.get("atr_min_period", 7) or 7),
                atr_max_period=int(st_cfg_dict.get("atr_max_period", 21) or 21),
                multiplier_min=float(st_cfg_dict.get("multiplier_min", 1.5) or 1.5),
                multiplier_max=float(st_cfg_dict.get("multiplier_max", 4.0) or 4.0),
                er_period=int(st_cfg_dict.get("er_period", 10) or 10),
                adx_period=int(st_cfg_dict.get("adx_period", 14) or 14),
                use_bb_correction=bool(st_cfg_dict.get("use_bb_correction", True)),
                bb_period=int(st_cfg_dict.get("bb_period", 20) or 20),
                bb_std=float(st_cfg_dict.get("bb_std", 2.0) or 2.0),
                smooth_period=int(st_cfg_dict.get("smooth_period", 3) or 3),
            )
            self._st = AdaptiveSuperTrend(self._st_cfg)
            logger.info("[ChartEngine] AdaptiveSuperTrend 초기화 완료")
        except Exception as e:
            logger.warning("[ChartEngine] AdaptiveSuperTrend 초기화 실패: %s", e)
            self._st = None
            self._st_cfg = None

    def _apply_supertrend_from_instance(
        self,
        df: pd.DataFrame,
        st: Any,
    ) -> pd.DataFrame:
        """[SSOT] 외부 주입된 AdaptiveSuperTrend 인스턴스에서 df 컬럼을 추출.

        추출 전략 (우선순위):
        1. compute_from_df(df) — AdaptiveSuperTrend가 DataFrame 재계산 API 제공 시
        2. 내부 히스토리 + 미완결봉 1봉 증분 — get_history() API 제공 시
        3. replay 폴백 — 위 두 방법 모두 불가 시

        외부 주입 인스턴스는 predictor가 이미 최신 상태로 유지하므로
        완결봉 히스토리를 직접 참조하는 방식이 가장 효율적이다.
        """
        try:
            # ── 전략 1: compute_from_df ──────────────────────────────────────
            compute_fn = getattr(st, 'compute_from_df', None)
            if callable(compute_fn):
                try:
                    result_df = compute_fn(df)
                    if "SuperTrend" in result_df.columns and "SuperTrend_Dir" in result_df.columns:
                        df["SuperTrend"]     = result_df["SuperTrend"].values
                        df["SuperTrend_Dir"] = result_df["SuperTrend_Dir"].values
                        logger.debug(
                            "[ChartEngine][SSOT] ST compute_from_df 적용: %d봉", len(df)
                        )
                        return df
                except Exception as e:
                    logger.debug("[ChartEngine][SSOT] compute_from_df 실패: %s", e)

            # ── 전략 2: 내부 히스토리 + 미완결봉 증분 ──────────────────────
            get_hist = getattr(st, 'get_history', None)
            if callable(get_hist):
                try:
                    history = get_hist()  # List[SuperTrendState]
                    n_df    = len(df)
                    if len(history) >= n_df - 1:
                        tail  = history[-(n_df - 1):]
                        vals  = [s.value     for s in tail]
                        dirs  = [s.direction for s in tail]
                        last  = df.iloc[-1]
                        live  = st.update(
                            float(last["High"]), float(last["Low"]), float(last["Close"])
                        )
                        df["SuperTrend"]     = vals + [live.value]
                        df["SuperTrend_Dir"] = dirs + [live.direction]
                        logger.debug(
                            "[ChartEngine][SSOT] ST history 적용: hist=%d df=%d",
                            len(history), n_df,
                        )
                        return df
                except Exception as e:
                    logger.debug("[ChartEngine][SSOT] history 적용 실패: %s", e)

        except Exception as e:
            logger.warning("[ChartEngine][SSOT] _apply_supertrend_from_instance 예외: %s", e)

        # ── 전략 3: replay 폴백 ─────────────────────────────────────────────
        logger.debug("[ChartEngine][SSOT] ST 폴백: replay 계산")
        return self._compute_supertrend_replay(df)

    def _compute_supertrend_replay(self, df: pd.DataFrame) -> pd.DataFrame:
        """SuperTrend replay 계산 (내부 self._st 인스턴스 사용).

        _apply_supertrend_from_instance의 폴백 경로.
        기존 compute() 내 캐시 미스 경로 로직을 메서드로 추출.
        _st_external=False(자체 관리) 경로에서도 동일하게 사용한다.
        """
        if self._st is None:
            return df
        try:
            now_floor = pd.Timestamp.now().floor("1min")
            completed = df[df.index < now_floor]
            if completed.empty:
                completed = df.iloc[:-1]

            n_completed = len(completed)
            n_fed       = self._st_fed_bars

            if n_fed == 0 or n_fed > n_completed:
                # 전체 재계산
                self._st.reset()
                st_values, st_dirs = [], []
                for _, row in completed.iterrows():
                    state = self._st.update(
                        float(row["High"]), float(row["Low"]), float(row["Close"])
                    )
                    st_values.append(state.value)
                    st_dirs.append(state.direction)
                logger.info("[ChartEngine] ST replay 전체 재계산: %d 완결봉", n_completed)
            else:
                # 증분 feed
                new_rows  = completed.iloc[n_fed:]
                st_values = list(self._st_cache_values or [])
                st_dirs   = list(self._st_cache_dirs   or [])
                for _, row in new_rows.iterrows():
                    state = self._st.update(
                        float(row["High"]), float(row["Low"]), float(row["Close"])
                    )
                    st_values.append(state.value)
                    st_dirs.append(state.direction)
                if new_rows.shape[0] > 0:
                    logger.info(
                        "[ChartEngine] ST replay 증분: +%d봉 (총 %d완결봉)",
                        new_rows.shape[0], n_completed,
                    )

            # 완결봉 캐시 저장
            sig = getattr(self, '_last_sig', None)
            self._st_cache_sig    = sig
            self._st_cache_values = list(st_values)
            self._st_cache_dirs   = list(st_dirs)
            self._st_fed_bars     = n_completed

            # 미완결봉 1봉 증분
            last_row   = df.iloc[-1]
            live_state = self._st.update(
                float(last_row["High"]), float(last_row["Low"]), float(last_row["Close"])
            )
            st_all   = st_values + [live_state.value]
            st_d_all = st_dirs   + [live_state.direction]

            # df 길이와 맞추기
            n_df = len(df)
            if len(st_all) == n_df:
                df["SuperTrend"]     = st_all
                df["SuperTrend_Dir"] = st_d_all
            elif len(st_all) < n_df:
                pad     = n_df - len(st_all)
                fv      = next((v for v in st_all if not np.isnan(v)), np.nan)
                fv_dir  = next((v for v in st_d_all if not np.isnan(v)), np.nan)
                df["SuperTrend"]     = [fv]    * pad + st_all
                df["SuperTrend_Dir"] = [fv_dir] * pad + st_d_all
            else:
                trimmed   = st_all[-n_df:]
                trimmed_d = st_d_all[-n_df:]
                if not np.all(np.isnan(trimmed)):
                    df["SuperTrend"]     = trimmed
                    df["SuperTrend_Dir"] = trimmed_d

        except Exception as e:
            logger.warning("[ChartEngine] _compute_supertrend_replay 실패: %s", e)

        return df

    def _init_zigzag(self, cfg: Optional[Any] = None, data_source: Optional[str] = None) -> None:
        """indicators.AdaptiveZigZag 인스턴스를 생성한다.

        [SSOT] set_zigzag()로 외부 주입된 인스턴스가 있으면 재생성하지 않는다.
        chart_engine 의 ZigZag 는 항상 predictor._adaptive_mgr 의 인스턴스를 SSOT 로
        사용해야 하므로, _zz_external=True 일 때는 아무것도 하지 않고 반환한다.
        단, data_source 가 변경된 경우(소스 전환)에는 _zz_external 을 리셋하고
        새 소스에 맞는 인스턴스를 다시 주입받을 때까지 내부 생성도 허용하지 않는다.
        """
        # [SSOT] 외부 주입 ZigZag 가 있으면 재생성 금지
        if self._zz_external and self._zz is not None:
            logger.debug(
                "[ChartEngine] _init_zigzag: 외부 주입 ZigZag 보존 (data_source=%s)",
                data_source,
            )
            return

        AdaptiveZigZag = None

        try:
            from indicators import AdaptiveZigZag
        except Exception as e:
            logger.warning("[ChartEngine] indicators import failed: %s — 피봇 마커 비활성", e)

        if AdaptiveZigZag is None:
            logger.warning("[ChartEngine] kospi_indicators 없음 — 피봇 마커 비활성")
            return

        # [SSOT] AdaptiveZigZagConfig 는 AdaptiveZigZagSettings.to_zigzag_config() 경유로 생성.
        # setattr 루프 직접 복사를 제거하고 AppConfig.adaptive_indicator 의 Settings 객체를
        # data_source 에 따라 선택한 뒤 to_zigzag_config() 를 호출한다.
        ds = data_source or self._current_data_source
        if cfg is not None:
            try:
                ai = cfg.adaptive_indicator
                if ds == "futures":
                    zz_s = ai.futures_zigzag or ai.zigzag
                elif ds == "kospi":
                    zz_s = ai.kospi_zigzag or ai.zigzag
                else:
                    zz_s = ai.zigzag
                # data_source 에 맞는 log prefix
                _prefix = {"futures": "[KP200]", "kospi": "[KOSPI]"}.get(ds or "", "")
                logger.info(
                    "[ChartEngine] _init_zigzag (SSOT): data_source=%s "
                    "use_atr=%s min_wave_pct=%s pivot_min_pct=%s",
                    ds,
                    getattr(zz_s, "use_atr_based_filtering", False),
                    getattr(zz_s, "min_wave_pct", None),
                    getattr(zz_s, "pivot_threshold_min_pct", None),
                )
                zz_cfg = zz_s.to_zigzag_config(
                    pivot_lifecycle_log=True,
                    pivot_lifecycle_log_prefix=_prefix,
                )
            except Exception as _e:
                logger.error(
                    "[ChartEngine] _init_zigzag: AppConfig 적용 실패 → 기본값 AdaptiveZigZagConfig() 사용. "
                    "config.json 설정이 차트에 반영되지 않습니다: %s", _e
                )
                from indicators.adaptive_zigzag import AdaptiveZigZagConfig
                zz_cfg = AdaptiveZigZagConfig()
        else:
            # cfg 없음(독립 실행 등) → 기본값으로 ZigZag 생성
            logger.warning(
                "[ChartEngine] _init_zigzag: cfg=None — 기본값 AdaptiveZigZagConfig() 사용 "
                "(config.json 미연결 상태)"
            )
            from indicators.adaptive_zigzag import AdaptiveZigZagConfig
            zz_cfg = AdaptiveZigZagConfig()

        self._zz     = AdaptiveZigZag(zz_cfg)
        self._zz_cfg = zz_cfg
        logger.info("[ChartEngine] AdaptiveZigZag 초기화 완료: _zz=%s", self._zz is not None)

        # 데이터 소스 변경 시 SuperTrend도 재초기화 (완전 재생성)
        self._init_supertrend(cfg)

        # 심볼 설정 (data_source에 따라)
        ds = data_source or self._current_data_source
        if ds == "kospi":
            self._zz.set_symbol("KOSPI 지수")
        elif ds == "futures":
            self._zz.set_symbol("KP200 선물")
        else:
            self._zz.set_symbol("KP200 선물")

        # 피봇 후보 콜백 설정
        if hasattr(self._zz, 'pivot_collector'):
            collector = self._zz.pivot_collector
            if collector is not None and self._pivot_candidate_callback is not None:
                collector.set_callback(self._pivot_candidate_callback, change_cooldown_sec=60.0)
                logger.info("[ChartEngine] 피봇 후보 콜백 설정 완료")
            elif collector is not None:
                logger.warning("[ChartEngine] 피봇 콜밭이 설정되지 않음 (pivot_candidate_callback=None)")
            else:
                logger.debug("[ChartEngine] 피봇 콜렉터가 비활성화됨 (enable_pivot_collector=False)")
        else:
            logger.debug("[ChartEngine] 피봇 콜렉터 속성 없음")

        # [PIVOT-EVENT-LOG] ZigZag 직접 피봇 이벤트 콜백 설정 (과거 데이터 replay 지원)
        if hasattr(self._zz, 'set_pivot_event_callback'):
            if self._pivot_candidate_callback is not None:
                self._zz.set_pivot_event_callback(self._pivot_candidate_callback)
                logger.info("[ChartEngine] ZigZag 직접 피봇 이벤트 콜백 설정 완료")
            else:
                logger.warning("[ChartEngine] 피봇 이벤트 콜밭이 설정되지 않음 (pivot_candidate_callback=None)")
        else:
            logger.warning("[ChartEngine] ZigZag에 set_pivot_event_callback 메서드 없음")

        # 레짐 기반 파라미터 조정 활성화 시 RegimeParamMapper 주입
        if self._adaptive_enabled:
            self._enable_regime_mapping()

    def _enable_regime_mapping(self) -> None:
        """RegimeParamMapper를 ZigZag에 주입하여 레짐 기반 파라미터 조정 활성화."""
        if self._zz is None:
            logger.warning("[ChartEngine] ZigZag 인스턴스 없음 - 레짐 매핑 비활성")
            return

        try:
            from indicators.regime_param_mapper import patch_zigzag_with_regime

            ds = self._current_data_source or "futures"
            symbol = "futures" if ds == "futures" else "kospi"

            self._regime_mapper = patch_zigzag_with_regime(
                self._zz,
                config=None,  # 기본 설정 사용
                symbol=symbol,
                classify_interval_bars=10,  # 10봉마다 레짐 재분류
            )

            logger.info("[ChartEngine] RegimeParamMapper 주입 완료: symbol=%s", symbol)
        except Exception as e:
            logger.error("[ChartEngine] RegimeParamMapper 주입 실패: %s", e, exc_info=True)
            self._adaptive_enabled = False

    def _reattach_regime_mapper(self) -> None:
        """ZigZag 재생성 후 regime_mapper를 새 인스턴스에 재주입한다.

        _feed_zigzag()에서 self._zz = self._zz.__class__(cfg) 로 인스턴스가
        교체될 때마다 호출해야 한다. mapper가 없거나 adaptive 비활성 상태면 무시.
        """
        if not self._adaptive_enabled or self._regime_mapper is None or self._zz is None:
            return
        try:
            self._zz._param_adjuster = self._regime_mapper
            self._zz.set_adaptive_enabled(True)
            logger.debug("[ChartEngine] regime_mapper 재주입 완료")
        except Exception as e:
            logger.warning("[ChartEngine] regime_mapper 재주입 실패: %s", e)

    def set_adaptive_enabled(self, enabled: bool) -> None:
        """Adaptive 모드 활성화/비활성화."""
        self._adaptive_enabled = enabled
        if enabled and self._zz is not None and self._regime_mapper is None:
            self._enable_regime_mapping()
        elif not enabled:
            self._regime_mapper = None
        logger.info("[ChartEngine] Adaptive 모드: %s", enabled)

    def get_current_regime(self) -> Optional[str]:
        """현재 시장 레짐 반환."""
        if self._regime_mapper is not None:
            return self._regime_mapper.stable_regime.value
        return None

    # ── [FIX-5] 데이터 서명 계산 ───────────────────────────────────────────────

    def _compute_data_signature(self, df: pd.DataFrame) -> str:
        """데이터프레임의 "고유 특성"을 해시 기반 서명으로 생성.

        DataFrame 길이와 무관하게, 동일한 봉 데이터면 동일 서명 반환.
        """
        import hashlib

        # ── 오늘 장중 완결봉만 추출 (09:00 ~ 마지막 완결봉) ──────────────
        try:
            now_floor = pd.Timestamp.now().floor("1min")
            completed = df[df.index < now_floor]
            if completed.empty:
                completed = df.iloc[:-1] if len(df) > 1 else df
        except Exception:
            completed = df.iloc[:-1] if len(df) > 1 else df

        if completed.empty:
            return "empty"

        # ── 컬럼 이름 대소문자 무시 처리 ─────────────────────────────────
        # 컬럼 이름이 'Close', 'close', 'CLOSE' 등 다양한 형태로 올 수 있음
        close_col = None
        for col in completed.columns:
            if col.lower() == 'close':
                close_col = col
                break
        if close_col is None:
            logger.warning("[ChartEngine] Close 컬럼을 찾을 수 없음: %s", completed.columns.tolist())
            return "no_close_column"
        # ────────────────────────────────────────────────────────────────────────────

        # 서명: 마지막 완결봉 시각 + 봉 수 + 마지막 종가
        key_str = (
            f"{completed.index[-1]}"
            f"_{len(completed)}"
            f"_{round(float(completed[close_col].iloc[-1]), 2)}"
        )
        signature = hashlib.md5(key_str.encode()).hexdigest()

        logger.debug("[ChartEngine] 데이터 서명: %s (봉 수: %d)", signature, len(completed))
        return signature

    # ────────────────────────────────────────────────────────────────────────────

    # ── ZigZag 전체 replay ───────────────────────────────────────────────────

    def _feed_zigzag(self, df: pd.DataFrame, exclude_last: bool = True, force_recompute: bool = False) -> None:
        """ZigZag 인스턴스를 초기화하고 df 전체 봉을 순차 replay한다.

        Args:
            df: OHLC 데이터
            exclude_last: 마지막 봉 제외 여부 (실시간 갱신 시 미완결 봉 포함 방지)
            force_recompute: 강제 재계산 여부 (캐시 무시)
        """
        logger.debug("[ChartEngine] _feed_zigzag 호출: df.shape=%s, exclude_last=%s, force_recompute=%s", df.shape, exclude_last, force_recompute)

        if self._zz is None:
            logger.error("[ChartEngine] _feed_zigzag: _zz가 None - _init_zigzag 호출")
            cfg = None
            try:
                from config import AppConfig
                cfg = AppConfig.from_file("config.json")
            except Exception as e:
                logger.warning("[ChartEngine] AppConfig 로드 실패: %s", e)
            self._init_zigzag(cfg, self._current_data_source)

        # [FIX] _zz_cfg가 None인 경우 _zz 인스턴스에서 config 가져오기
        if self._zz_cfg is None and self._zz is not None and hasattr(self._zz, 'config'):
            self._zz_cfg = self._zz.config
            logger.info("[ChartEngine] _zz_cfg를 _zz 인스턴스에서 설정 완료")
        
        if self._zz is None or self._zz_cfg is None:
            logger.warning("[ChartEngine] _feed_zigzag: _zz 또는 _zz_cfg가 None")
            return
        
        # ── [보완-2] 최소 데이터 조건 검사 (장 초반 데이터 부족 시 오류 방지) ──
        # ATR 기간 + 확인 봉수 + 여유분(5봉) 이상 필요
        min_required_bars = getattr(self._zz_cfg, 'atr_period', 14) + getattr(self._zz_cfg, 'confirmation_bars', 1) + 5
        if len(df) < min_required_bars:
            logger.warning("[ChartEngine] 데이터 부족으로 ZigZag 계산 스킵: %d봉 (필요 %d봉 이상)", len(df), min_required_bars)
            return
        # ────────────────────────────────────────────────────────────────────────────

        # ── [FIX-5] 데이터 서명 계산 ──────────────────────────────────────────────
        sig = self._compute_data_signature(df)

        # ── 캐시된 피봇 타입 검증: 모든 피봇이 동일한 타입이면 캐시 삭제 ──
        cache_invalidated = False
        if self._confirmed_pivots_cache:
            cached_types = []
            for cached_pivot in self._confirmed_pivots_cache:
                sw_type = cached_pivot["swing_type"]
                # swing_type은 enum 문자열("high"/"low"), 대문자("HIGH"/"LOW"), 또는 전체 문자열("SwingType.HIGH"/"SWINGTYPE.HIGH")일 수 있음
                sw_type_upper = str(sw_type).upper()
                if "HIGH" in sw_type_upper or sw_type_upper == "H":
                    sw_type_str = "H"
                else:
                    sw_type_str = "L"
                cached_types.append(sw_type_str)
            
            if cached_types and all(t == cached_types[0] for t in cached_types):
                logger.warning("[ChartEngine] 캐시된 피봇이 모두 동일한 타입(%s)입니다. 캐시 삭제 후 재계산합니다.", cached_types[0])
                self._confirmed_pivots_cache = []  # [FIX] None → [] 타입 통일
                # ZigZag 상태 캐시도 삭제하여 강제 재계산
                if sig in self._zz_state_cache:
                    del self._zz_state_cache[sig]
                self._replay_signature = None
                cache_invalidated = True
                # [FIX] 외부 주입 인스턴스는 교체 금지 (origin 보존 필수)
                if getattr(self, '_zz_external', False):
                    try:
                        self._zz.reset()
                    except AttributeError:
                        pass
                    logger.debug("[ChartEngine] ZigZag 외부 인스턴스 reset (동일타입 캐시 무효화)")
                else:
                    try:
                        self._zz = self._zz.__class__(self._zz_cfg)
                        logger.debug("[ChartEngine] ZigZag 인스턴스 재생성 완료")
                        self._reattach_regime_mapper()
                    except Exception as e:
                        logger.debug("[ChartEngine] ZigZag 인스턴스 재생성 오류: %s", e)
                # [BUG-FIX] return 제거: 캐시 삭제 후 아래 replay로 자연스럽게 진행
                # 기존 return은 ZigZag 재생성 후 replay 없이 종료되어 _all_swings=[] 상태를 유발했음.
                # _replay_signature=None으로 이미 초기화됐으므로 캐시 히트 없이 정상 replay 진행됨.
        # ────────────────────────────────────────────────────────────────────────────

        # 캐시 확인: 동일 서명이면 이전 ZigZag 상태 재사용
        # force_recompute=True이면 캐시 무시하고 강제 replay
        if not force_recompute and sig == self._replay_signature and sig in self._zz_state_cache:
            logger.info("[ChartEngine] ZigZag 캐시 히트: %s (skipped replay)", sig)
            cached_state = self._zz_state_cache[sig]
            self._zz = cached_state['zz_instance']
            self._confirmed_pivots_cache = cached_state['confirmed_pivots'].copy()  # [BUG-4] 복사본 사용
            # [BUG-4] 캐시 히트 시 _enforce_hl_alternation 호출 제거
            # 호출하면 _all_swings가 변경되어 캐시 불일치 발생
            # 캐시된 상태를 그대로 사용하여 일관성 유지
            return
        # ────────────────────────────────────────────────────────────────────────

        # 새 replay 필요
        self._replay_signature = sig

        try:
            # __init__ 직접 호출 대신 새 인스턴스 생성 (안전한 reset)
            if self._zz is None:
                logger.error("[ChartEngine] ZigZag 인스턴스 재생성 실패: _zz가 None")
                return
            if getattr(self, '_zz_external', False):
                # [FIX] 외부 주입 인스턴스는 교체 금지 — reset()으로만 초기화
                # 교체 시 _prepare_refresh의 동일성 체크가 깨져 set_zigzag 무한 루프 발생
                origin = getattr(self, '_zz_external_origin', None)
                if origin is not None:
                    self._zz = origin  # 원본 복원
                try:
                    self._zz.reset()
                except AttributeError:
                    pass  # reset() 없는 구형 인스턴스 허용
                logger.info("[ChartEngine] ZigZag 외부 인스턴스 reset 완료: _zz=%s", self._zz is not None)
                # [PIVOT-EVENT-LOG] reset 후 콜백 재설정
                if hasattr(self._zz, 'set_pivot_event_callback') and self._pivot_candidate_callback is not None:
                    self._zz.set_pivot_event_callback(self._pivot_candidate_callback)
                    logger.info("[ChartEngine] reset 후 ZigZag 피봇 이벤트 콜백 재설정 완료")
                # [PIVOT-EVENT-LOG] reset 후 심볼 재설정
                ds = self._current_data_source or "futures"
                if ds == "kospi":
                    self._zz.set_symbol("KOSPI 지수")
                elif ds == "futures":
                    self._zz.set_symbol("KP200 선물")
                else:
                    self._zz.set_symbol("KP200 선물")
            else:
                self._zz = self._zz.__class__(self._zz_cfg)
                logger.info("[ChartEngine] ZigZag 인스턴스 재생성 완료: _zz=%s", self._zz is not None)
                # [PIVOT-EVENT-LOG] 재생성 후 콜백 설정
                if hasattr(self._zz, 'set_pivot_event_callback') and self._pivot_candidate_callback is not None:
                    self._zz.set_pivot_event_callback(self._pivot_candidate_callback)
                    logger.info("[ChartEngine] 재생성 후 ZigZag 피봇 이벤트 콜백 설정 완료")
            # [FIX] 재생성/reset 후 regime_mapper 재주입
            self._reattach_regime_mapper()
        except Exception as e:
            logger.error("[ChartEngine] ZigZag 초기화 오류: %s", e, exc_info=True)
            return

        # ── 오늘 장중 완결봉만 replay ──────────────────────────────────
        try:
            now_floor = pd.Timestamp.now().floor("1min")
            rows = df[df.index < now_floor]
            logger.debug("[ChartEngine] 오늘 장중 필터링: now_floor=%s, df.shape=%s, rows.shape=%s", 
                        now_floor, df.shape, rows.shape)
            if rows.empty:
                rows = df.iloc[:-1] if len(df) > 1 else df
                logger.debug("[ChartEngine] rows.empty - 마지막 봉 제외: rows.shape=%s", rows.shape)
        except Exception as e:
            logger.warning("[ChartEngine] 오늘 장중 필터링 오류: %s", e)
            rows = df.iloc[:-1] if len(df) > 1 else df

        logger.debug("[ChartEngine] 필터링 후 rows.shape=%s, exclude_last=%s", rows.shape, exclude_last)

        # [BUG-FIX] exclude_last 파라미터 실제 반영:
        # now_floor 필터가 과거 데이터 마지막 봉을 걸러내지 못하는 경우(rows==df)
        # exclude_last=True이면 마지막 봉(미완결 가능성)을 명시적으로 제외.
        if exclude_last and len(rows) == len(df) and len(rows) > 1:
            rows = rows.iloc[:-1]
        # ────────────────────────────────────────────────────────────────

        # ── 컬럼 이름 대소문자 무시 처리 ─────────────────────────────────
        # 컬럼 이름이 'High', 'Low', 'Close', 'Open' 등 다양한 형태로 올 수 있음
        col_map = {}
        for col in rows.columns:
            col_lower = col.lower()
            if col_lower in ['high', 'low', 'close', 'open']:
                col_map[col_lower] = col

        # SuperTrend 방향 컬럼 확인
        st_dir_col = None
        for col in rows.columns:
            col_lower = col.lower()
            if col_lower in ['supertrend_dir', 'supertrenddirection']:
                st_dir_col = col
                break

        if not all(key in col_map for key in ['high', 'low', 'close', 'open']):
            missing = [k for k in ['high', 'low', 'close', 'open'] if k not in col_map]
            logger.warning("[ChartEngine] 필수 OHLC 컬럼을 찾을 수 없음: %s (가용: %s)",
                         missing, rows.columns.tolist())
            return
        # ────────────────────────────────────────────────────────────────────────────

        # [SUPERTREND-INTEGRATION] SuperTrend 필터 활성화 확인
        supertrend_pivot_filter = False
        try:
            from config import AppConfig
            cfg = AppConfig.from_file("config.json")
            # supertrend_pivot_filter는 adaptive_indicator 섹션에 있음
            adaptive_indicator = getattr(cfg, 'adaptive_indicator', None)
            if adaptive_indicator is not None:
                supertrend_pivot_filter = getattr(adaptive_indicator, 'supertrend_pivot_filter', False)
        except Exception as e:
            logger.debug("[ChartEngine] supertrend_pivot_filter 로드 실패: %s", e)

        logger.info("[ChartEngine][ST] SuperTrend 필터 상태: supertrend_pivot_filter=%s, st_dir_col=%s",
                    supertrend_pivot_filter, st_dir_col)

        # numpy 배열 직접 접근 (성능 최적화)
        highs = rows[col_map['high']].to_numpy(dtype=np.float64)
        lows = rows[col_map['low']].to_numpy(dtype=np.float64)
        closes = rows[col_map['close']].to_numpy(dtype=np.float64)
        opens = rows[col_map['open']].to_numpy(dtype=np.float64)
        times = rows.index

        # SuperTrend 방향 배열 (존재하면)
        st_dirs = None
        if st_dir_col is not None and supertrend_pivot_filter:
            st_dirs = rows[st_dir_col].to_numpy(dtype=np.float64)
            logger.info("[ChartEngine][ST] SuperTrend_Dir 컬럼 로드: len=%d", len(st_dirs))
        else:
            logger.warning("[ChartEngine][ST] SuperTrend_Dir 컬럼 없거나 필터 비활성화: st_dir_col=%s, filter=%s",
                         st_dir_col, supertrend_pivot_filter)

        _feed_t0 = _time.perf_counter() if "time" in dir() else None
        try:
            import time as _time
            _feed_t0 = _time.perf_counter()
        except Exception:
            _feed_t0 = None
        logger.info("[ChartEngine][RT] ZigZag replay 시작: %d봉 (exclude_last=%s)", len(rows), exclude_last)
        for i in range(len(rows)):
            try:
                h, l, c, o = highs[i], lows[i], closes[i], opens[i]
                if h <= 0 or l <= 0 or c <= 0:
                    continue
                bt = times[i] if isinstance(times[i], pd.Timestamp) else pd.Timestamp(times[i])
                # datetime 객체로 변환하여 전달
                if isinstance(bt, pd.Timestamp):
                    bt = bt.to_pydatetime()
                
                # [FIX] 4번째 봉에서 시가 anchor 심기 - 장 초반 잘못된 피봇 방지
                # 심볼별 장 시작 시간 고려: KP200 08:45부터, KOSPI 09:00부터
                if i == 3 and o > 0 and hasattr(self._zz, 'seed_anchor'):
                    try:
                        from indicators.adaptive_zigzag import SwingType
                        # 장 시작 시간 확인
                        symbol = getattr(self._zz, '_symbol_name', '')
                        market_start_hour = 9
                        market_start_minute = 0
                        if "KP200" in symbol or "선물" in symbol:
                            market_start_hour = 8
                            market_start_minute = 45
                        
                        # 현재 봉 시간 확인
                        bar_hour = bt.hour if hasattr(bt, 'hour') else 0
                        bar_minute = bt.minute if hasattr(bt, 'minute') else 0
                        
                        # [FIX] 장 시작 시간 확인 완화 - 4번째 봉에서는 무조건 anchor 심기
                        # KP200은 장 시작이 08:45이지만, 4번째 봉이 08:45 이전이어도 anchor 심기 허용
                        # 장 시작 시간 확인은 pivot 생성 시에만 적용
                        # anchor 심기는 장 시작 시간과 무관하게 허용
                        # if bar_hour > market_start_hour or (bar_hour == market_start_hour and bar_minute >= market_start_minute):
                        # [FIX] 시가 anchor 타입 결정 개선
                        # 첫 4봉의 고가/저가 중 시가보다 더 큰 움직임 쪽을 anchor로 선택
                        # 시가 대비 상승 폭이 하락 폭보다 크면 HIGH anchor
                        # 시가 대비 하락 폭이 상승 폭보다 크면 LOW anchor
                        max_high = max(highs[:4]) if len(highs) >= 4 else h
                        min_low = min(lows[:4]) if len(lows) >= 4 else l
                        up_move = max_high - o
                        down_move = o - min_low
                        anc_type = SwingType.HIGH if up_move >= down_move else SwingType.LOW
                        self._zz.seed_anchor(o, anc_type)
                        logger.debug("[ChartEngine] seed_anchor 주입 (4번째 봉): %.2f, type=%s (up=%.2f, down=%.2f, 장 시작: %02d:%02d, 봉시간: %02d:%02d)", o, anc_type, up_move, down_move, market_start_hour, market_start_minute, bar_hour, bar_minute)
                    except Exception as e:
                        logger.warning("[ChartEngine] seed_anchor 주입 실패: %s", e)

                # [SUPERTREND-INTEGRATION] SuperTrend 신호 전달
                if supertrend_pivot_filter and st_dirs is not None and i < len(st_dirs):
                    st_dir = st_dirs[i]
                    if hasattr(self._zz, 'set_supertrend_signal'):
                        st_signal = "bull" if st_dir == 1 else ("bear" if st_dir == -1 else "")
                        if st_signal:
                            self._zz.set_supertrend_signal(st_signal)
                            if i == 0 or i == len(rows) - 1:
                                logger.debug("[ChartEngine][ST] SuperTrend 신호 전달: idx=%d, time=%s, signal=%s, dir=%s",
                                            i, bt, st_signal, st_dir)

                self._zz.update(high=h, low=l, close=c, open=o, bar_time=bt)
                if i == 0 or i == len(rows) - 1:
                    logger.debug("[ChartEngine] ZigZag update: idx=%d, time=%s, h=%.2f, l=%.2f, c=%.2f",
                                i, bt, h, l, c)
            except Exception as e:
                logger.error("[ChartEngine] 봉 업데이트 오류 (idx=%d): %s", i, e, exc_info=True)
                continue

        _feed_elapsed = (_time.perf_counter() - _feed_t0) if _feed_t0 else 0
        _swing_cnt = len(list(getattr(self._zz, "_all_swings", []) or []))
        logger.info(
            "[ChartEngine][RT] ZigZag replay 완료: %d봉 → swing=%d elapsed=%.3fs",
            len(rows), _swing_cnt, _feed_elapsed,
        )

        # [H/L-교번-사후처리] 데이터 로드 완료 후 H/L 교번 강제
        if hasattr(self._zz, "_enforce_hl_alternation"):
            self._zz._enforce_hl_alternation()

        # 캐시 저장 (데이터가 최소 20봉 이상일 때만 저장)
        swings = list(getattr(self._zz, "_all_swings", []) or [])
        # [FIX] 데이터 부족 시 피봇 캐시 저장 방지: 최소 20봉 이상일 때만 저장
        min_bars_for_cache = 20
        if len(df) >= min_bars_for_cache:
            self._confirmed_pivots_cache = [
                {
                    "index": sw.index,
                    "price": float(sw.price),
                    "swing_type": str(sw.swing_type).upper(),  # 대문자로 저장하여 일관성 유지
                    "confirmed": bool(getattr(sw, "confirmed", False)),
                    "confirmed_at_idx": getattr(sw, "confirmed_at_idx", sw.index),  # [BUG-2] confirmed_at_idx 추가
                }
                for sw in swings
                if getattr(sw, "confirmed", False)
            ]
        else:
            logger.debug("[ChartEngine] 데이터 부족(%d봉 < %d봉)으로 피봇 캐시 저장 스킵", len(df), min_bars_for_cache)

        # ── [FIX-7] ZigZag 상태 캐시 저장 (캐시 무효화된 경우 제외) ───────────────
        # [FIX] 데이터 부족 시 ZigZag 상태 캐시도 저장하지 않음
        if not cache_invalidated and len(df) >= min_bars_for_cache:
            # [DESIGN-3] ZigZag 인스턴스 딥카피로 공유 참조 문제 해결
            # [PIVOT-EVENT-LOG] 콜백 함수는 deepcopy할 수 없으므로 일시적으로 제거
            import copy
            pivot_callback = None
            if hasattr(self._zz, '_pivot_event_callback'):
                pivot_callback = self._zz._pivot_event_callback
                self._zz._pivot_event_callback = None
            
            try:
                self._zz_state_cache[sig] = {
                    'zz_instance': copy.deepcopy(self._zz),  # 딥카피로 공유 참조 방지
                    'confirmed_pivots': self._confirmed_pivots_cache.copy(),
                    'timestamp': pd.Timestamp.now(),  # 캐시 정리용 타임스탬프
                }
                logger.debug("[ChartEngine] ZigZag 상태 캐시 저장: %s (캐시 크기: %d)", sig, len(self._zz_state_cache))
            finally:
                # 콜백 복원
                if pivot_callback is not None and hasattr(self._zz, '_pivot_event_callback'):
                    self._zz._pivot_event_callback = pivot_callback

            # 캐시 크기 제한 (최근 10개 상태만 유지)
            if len(self._zz_state_cache) > 10:
                try:
                    oldest_key = min(
                        self._zz_state_cache.keys(),
                        key=lambda k: self._zz_state_cache[k]['timestamp']
                    )
                    del self._zz_state_cache[oldest_key]
                    logger.debug("[ChartEngine] 가장 오래된 캐시 제거: %s", oldest_key)
                except (KeyError, ValueError) as e:
                    logger.warning("[ChartEngine] ZigZag 캐시 정리 실패, 전체 초기화: %s", e)
                    self._zz_state_cache = {}  # 실패 시 전체 초기화

            logger.debug("[ChartEngine] ZigZag 상태 캐시 저장: %s (캐시 크기: %d)", sig, len(self._zz_state_cache))
        # ────────────────────────────────────────────────────────────────────────

    # ── pivot_markers 딕셔너리 (SkyPlot 호환) ───────────────────────────────

    def _build_pivot_markers(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        ZigZag._all_swings → SkyPlot 호환 pivot_markers 딕셔너리.

        형식:
          {
            "confirmed":   {"idx": [int, …], "y": [float, …], "type": ["H"|"L", …], "confirmed_at_idx": [int, …], "registered_at_idx": [int, …]},
            "unconfirmed": {"idx": [int, …], "y": [float, …], "type": ["H"|"L", …], "registered_at_idx": [int, …]},
            "anchor_idx":  int   # -1 이면 없음
          }
        idx 는 df 행 번호(0-based).
        confirmed_at_idx는 확정된 봉의 인덱스 (확정되지 않은 경우 -1).
        registered_at_idx는 후보 등록된 봉의 인덱스 (차트 표시용).
        """
        logger.debug("[ChartEngine] _build_pivot_markers 호출: _zz=%s", self._zz is not None)
        import time as _time
        _pm_t0 = _time.perf_counter()
        
        if self._zz is None:
            logger.warning("[ChartEngine] _build_pivot_markers: _zz가 None")
            return None

        swings: List[Any] = list(getattr(self._zz, "_all_swings", []) or [])
        logger.debug("[ChartEngine] _build_pivot_markers: swings=%d", len(swings))
        
        # [DEBUG] ZigZag _all_swings 상세 출력
        if swings:
            swing_details = []
            for sw in swings:
                sw_idx = sw.index
                sw_price = sw.price
                sw_type = getattr(sw, "swing_type", "?")
                is_conf = getattr(sw, "confirmed", False)
                if 0 <= sw_idx < len(df):
                    ts = df.index[sw_idx]
                    time_str = str(ts).split(" ")[1][:5] if len(str(ts).split(" ")) > 1 else "?"
                else:
                    time_str = "?"
                swing_details.append(f"{sw_type}@{time_str}={sw_price:.2f}(conf={is_conf})")
            logger.info("[ChartEngine][ZZ] ZigZag _all_swings: %s", ", ".join(swing_details))
        
        if not swings:
            # 캐시된 확정 피봇만 있는 경우에도 반환
            if self._confirmed_pivots_cache:
                logger.debug("[ChartEngine] _build_pivot_markers: 캐시된 피봇 사용: %d", len(self._confirmed_pivots_cache))
                pass  # 캐시 처리 로직으로 이동
            else:
                logger.warning("[ChartEngine] _build_pivot_markers: swings 비어있고 캐시도 없음")
                return None

        # timestamp → row-index 매핑 테이블 (O(1) 조회)
        ts_idx = df.index
        ts_map: Dict[pd.Timestamp, int] = {}
        for i, t in enumerate(ts_idx):
            try:
                ts_map[pd.Timestamp(t)] = i
            except Exception as e:
                logger.debug("[ChartEngine] 타임스탬프 매핑 오류 (i=%d): %s", i, e)
                pass

        # SwingType.HIGH 판별 — import 실패 시 문자열 비교로 fallback
        _HIGH: Any = None
        for _import_path in (
            ("kospi_indicators",),
            ("kospi_indicators", "kospi_indicators", "adaptive_zigzag"),
        ):
            try:
                mod = __import__(".".join(_import_path), fromlist=["SwingType"])
                _HIGH = getattr(mod, "SwingType").HIGH
                break
            except Exception as e:
                logger.debug("[ChartEngine] SwingType import 실패 (%s): %s", _import_path, e)
                continue

        pm: Dict[str, Any] = {
            "confirmed":   {"idx": [], "y": [], "type": [], "confirmed_at_idx": [], "registered_at_idx": []},
            "unconfirmed": {"idx": [], "y": [], "type": [], "registered_at_idx": []},
            "anchor_idx": -1,
            "minutes": self._minutes,  # 범위 변경 시 피봇 마커 재렌더용
        }

        # 루프 전 numpy 배열 추출 (성능 최적화)
        highs_arr = df["High"].to_numpy(dtype=np.float64)
        lows_arr = df["Low"].to_numpy(dtype=np.float64)

        # ── 캐시된 확정 피봇 복원 ─────────────────────────────────────────────────
        # 이미 확정된 피봇은 캐시에서 복원하여 일관성 유지
        cached_confirmed_indices = set()  # 중복 방지용 인덱스 집합
        if self._confirmed_pivots_cache:
            for cached_pivot in self._confirmed_pivots_cache:
                try:
                    sw_idx = cached_pivot["index"]
                    sw_price = cached_pivot["price"]
                    sw_type = cached_pivot["swing_type"]

                    # 인덱스 매핑
                    if isinstance(sw_idx, (int, np.integer)):
                        bar_i = int(sw_idx)
                    else:
                        ts = pd.Timestamp(sw_idx)
                        bar_i = ts_map.get(ts, -1)
                        if bar_i < 0:
                            bar_i = _nearest_bar_idx(ts_idx, ts, tol_sec=90)

                    if bar_i < 0 or bar_i >= len(df):
                        continue  # 범위 밖이면 스킵

                    # ── [보완-5] ZigZag 피봇 타입 판별 안정성 (Enum vs String) ──
                    # swing_type은 enum 문자열("high"/"low"), 대문자("HIGH"/"LOW"), 
                    # 전체 문자열("SwingType.HIGH"/"SWINGTYPE.HIGH"), 또는 정수(0, 1)일 수 있음
                    try:
                        sw_type_upper = str(sw_type).upper()
                        if "HIGH" in sw_type_upper or sw_type_upper == "H" or sw_type_upper == "1":
                            sw_type_str = "H"
                        elif "LOW" in sw_type_upper or sw_type_upper == "L" or sw_type_upper == "0":
                            sw_type_str = "L"
                        else:
                            # 가격 기반 폴백: 이전 피봇 가격과 비교
                            if len(highs_arr) > bar_i and len(lows_arr) > bar_i:
                                prev_high = highs_arr[max(0, bar_i - 1)] if bar_i > 0 else highs_arr[0]
                                prev_low = lows_arr[max(0, bar_i - 1)] if bar_i > 0 else lows_arr[0]
                                if sw_price > prev_high:
                                    sw_type_str = "H"
                                elif sw_price < prev_low:
                                    sw_type_str = "L"
                                else:
                                    sw_type_str = "H"  # 기본값
                            else:
                                sw_type_str = "H"  # 기본값
                    except Exception:
                        sw_type_str = "H"  # 기본값
                    # ────────────────────────────────────────────────────────────────────────────
                    logger.debug("[ChartEngine] 캐시된 피봇: idx=%s price=%.2f type=%s (raw=%s)", sw_idx, sw_price, sw_type_str, sw_type)

                    # 피봇 마커 위치: 캐시된 피봇 가격 사용 (sw_price)
                    price = sw_price

                    # 캐시된 확정 피봇 추가
                    pm["confirmed"]["idx"].append(bar_i)
                    pm["confirmed"]["y"].append(price)
                    pm["confirmed"]["type"].append(sw_type_str)
                    # [BUG-2] 캐시된 confirmed_at_idx 사용 (기본값: bar_i)
                    cached_confirmed_at_idx = cached_pivot.get("confirmed_at_idx", bar_i)
                    # confirmed_at_idx도 인덱스 매핑 필요
                    if isinstance(cached_confirmed_at_idx, (int, np.integer)):
                        confirmed_bar_i = int(cached_confirmed_at_idx)
                    else:
                        ts = pd.Timestamp(cached_confirmed_at_idx)
                        confirmed_bar_i = ts_map.get(ts, -1)
                        if confirmed_bar_i < 0:
                            confirmed_bar_i = _nearest_bar_idx(ts_idx, ts, tol_sec=90)
                    pm["confirmed"]["confirmed_at_idx"].append(confirmed_bar_i if confirmed_bar_i >= 0 else bar_i)
                    pm["confirmed"]["registered_at_idx"].append(bar_i)

                    cached_confirmed_indices.add(bar_i)

                    # anchor 처리
                    if bar_i == 0:
                        pm["anchor_idx"] = 0

                except Exception as e:
                    logger.debug("[ChartEngine] 캐시된 피봇 복원 오류: %s", e)
                    continue

            if self._confirmed_pivots_cache:
                logger.debug("[ChartEngine] 캐시된 확정 피봇 복원: %d개", len(self._confirmed_pivots_cache))
        # ────────────────────────────────────────────────────────────────────────

        for sw in swings:
            try:
                sw_idx = sw.index   # int(bar_idx) 또는 Timestamp
                if isinstance(sw_idx, (int, np.integer)):
                    bar_i = int(sw_idx)
                    if not (0 <= bar_i < len(df)):
                        continue
                else:
                    ts    = pd.Timestamp(sw_idx)
                    bar_i = ts_map.get(ts, -1)
                    if bar_i < 0:
                        bar_i = _nearest_bar_idx(ts_idx, ts, tol_sec=90)
                    if bar_i < 0:
                        continue

                # swing_type 변환: enum 객체 또는 문자열 처리
                sw_type_raw = getattr(sw, "swing_type", None)
                if sw_type_raw is None:
                    sw_type_str = "L"  # 기본값
                    logger.debug("[ChartEngine] swing_type is None, defaulting to L")
                elif hasattr(sw_type_raw, "value"):
                    # enum 객체인 경우
                    sw_type_str = "H" if sw_type_raw.value in ("high", "HIGH", "H") else "L"
                    logger.debug("[ChartEngine] swing_type enum: value=%s, converted=%s", sw_type_raw.value, sw_type_str)
                else:
                    # 문자열인 경우
                    sw_type_str = "H" if str(sw_type_raw).upper() in ("HIGH", "H", "HIGH") else "L"
                    logger.debug("[ChartEngine] swing_type string: raw=%s, converted=%s", str(sw_type_raw), sw_type_str)

                # 피봇 마커 위치: swing.price 사용 (ZigZag가 계산한 정확한 피봇 가격)
                price = float(sw.price)

                is_conf = bool(getattr(sw, "confirmed", False))
                bucket  = "confirmed" if is_conf else "unconfirmed"

                # 캐시된 확정 피봇과 중복 방지
                if is_conf and bar_i in cached_confirmed_indices:
                    continue  # 이미 캐시된 확정 피봇이면 스킵

                # anchor: bar_i==0 인 confirmed 피봇 (시가 앵커)
                if bar_i == 0 and is_conf:
                    pm["anchor_idx"] = 0

                pm[bucket]["idx"].append(bar_i)
                pm[bucket]["y"].append(price)
                pm[bucket]["type"].append(sw_type_str)

                # 확정 시점 정보 추가
                if is_conf:
                    confirmed_at_idx = getattr(sw, "confirmed_at_idx", -1)
                    if isinstance(confirmed_at_idx, (int, np.integer)):
                        confirmed_at_bar_i = int(confirmed_at_idx)
                    else:
                        # Timestamp인 경우 매핑
                        conf_ts = pd.Timestamp(confirmed_at_idx)
                        confirmed_at_bar_i = ts_map.get(conf_ts, -1)
                        if confirmed_at_bar_i < 0:
                            confirmed_at_bar_i = _nearest_bar_idx(ts_idx, conf_ts, tol_sec=90)
                    pm[bucket]["confirmed_at_idx"].append(confirmed_at_bar_i if confirmed_at_bar_i >= 0 else -1)

                # 후보 등록 시점 정보 추가
                registered_at_idx = getattr(sw, "registered_at_idx", bar_i)
                if isinstance(registered_at_idx, (int, np.integer)):
                    registered_at_bar_i = int(registered_at_idx)
                else:
                    # Timestamp인 경우 매핑
                    reg_ts = pd.Timestamp(registered_at_idx)
                    registered_at_bar_i = ts_map.get(reg_ts, -1)
                    if registered_at_bar_i < 0:
                        registered_at_bar_i = _nearest_bar_idx(ts_idx, reg_ts, tol_sec=90)
                pm[bucket]["registered_at_idx"].append(registered_at_bar_i if registered_at_bar_i >= 0 else bar_i)

            except Exception as e:
                logger.debug("[ChartEngine] 피봇 마커 처리 오류: %s", e)
                continue

        # bar_idx 기준 정렬 (시간순)
        for bucket in ("confirmed", "unconfirmed"):
            if pm[bucket]["idx"]:
                if bucket == "confirmed":
                    # confirmed는 confirmed_at_idx, registered_at_idx도 함께 정렬
                    combined = list(zip(pm[bucket]["idx"], pm[bucket]["y"], pm[bucket]["type"], pm[bucket]["confirmed_at_idx"], pm[bucket]["registered_at_idx"]))
                    combined.sort(key=lambda x: x[0])  # bar_idx 기준 정렬
                    # [C-1] 명시적 언패킹으로 길이 불일치 방지
                    if combined:
                        result = list(zip(*combined))
                        pm[bucket]["idx"]              = list(result[0])
                        pm[bucket]["y"]                = list(result[1])
                        pm[bucket]["type"]             = list(result[2])
                        pm[bucket]["confirmed_at_idx"] = list(result[3])
                        pm[bucket]["registered_at_idx"]= list(result[4])
                    else:
                        pm[bucket]["idx"] = pm[bucket]["y"] = pm[bucket]["type"] = \
                        pm[bucket]["confirmed_at_idx"] = pm[bucket]["registered_at_idx"] = []
                else:
                    # unconfirmed는 registered_at_idx도 함께 정렬
                    combined = list(zip(pm[bucket]["idx"], pm[bucket]["y"], pm[bucket]["type"], pm[bucket]["registered_at_idx"]))
                    combined.sort(key=lambda x: x[0])  # bar_idx 기준 정렬
                    # [C-1] 명시적 언패킹으로 길이 불일치 방지
                    if combined:
                        result = list(zip(*combined))
                        pm[bucket]["idx"] = list(result[0])
                        pm[bucket]["y"] = list(result[1])
                        pm[bucket]["type"] = list(result[2])
                        pm[bucket]["registered_at_idx"] = list(result[3])
                    else:
                        pm[bucket]["idx"] = []
                        pm[bucket]["y"] = []
                        pm[bucket]["type"] = []
                        pm[bucket]["registered_at_idx"] = []

        # ── H/L 교번 강제 적용 (후처리) ──────────────────────────────────────────
        # 정렬된 피봇에서 연속된 H나 L이 나오면 제거하여 H/L 교번 보장
        # 주의: 너무 과도하게 제거하면 정상적인 피봇까지 손실될 수 있음
        # 현재는 비활성화하여 ZigZag 엔진의 _enforce_hl_alternation() 메서드에 의존
        # ZigZag 엔진은 이미 H/L 교번을 강제 적용하므로 후처리 불필요
        # for bucket in ("confirmed", "unconfirmed"):
        #     if not pm[bucket]["idx"]:
        #         continue
        #
        #     filtered_idx = []
        #     filtered_y = []
        #     filtered_type = []
        #     if bucket == "confirmed":
        #         filtered_confirmed_at = []
        #         filtered_registered_at = []
        #     else:
        #         filtered_registered_at = []
        #
        #     last_type = None
        #     for i, pivot_type in enumerate(pm[bucket]["type"]):
        #         # 연속된 동일 타입 제거
        #         if pivot_type == last_type:
        #             continue
        #         last_type = pivot_type
        #
        #         filtered_idx.append(pm[bucket]["idx"][i])
        #         filtered_y.append(pm[bucket]["y"][i])
        #         filtered_type.append(pivot_type)
        #         if bucket == "confirmed":
        #             filtered_confirmed_at.append(pm[bucket]["confirmed_at_idx"][i])
        #             filtered_registered_at.append(pm[bucket]["registered_at_idx"][i])
        #         else:
        #             filtered_registered_at.append(pm[bucket]["registered_at_idx"][i])
        #
        #     pm[bucket]["idx"] = filtered_idx
        #     pm[bucket]["y"] = filtered_y
        #     pm[bucket]["type"] = filtered_type
        #     if bucket == "confirmed":
        #         pm[bucket]["confirmed_at_idx"] = filtered_confirmed_at
        #         pm[bucket]["registered_at_idx"] = filtered_registered_at
        #     else:
        #         pm[bucket]["registered_at_idx"] = filtered_registered_at
        # ────────────────────────────────────────────────────────────────────────

        # [REVIEW-FIX-5] swing_version 추가: 클러스터링 in-place 갱신 시 캐시 무효화
        pm["swing_version"] = getattr(self._zz, "_swing_version", 0)
        _pm_elapsed = (_time.perf_counter() - _pm_t0)
        
        # [DEBUG] 빌드된 피봇 정보 상세 출력
        confirmed_count = len(pm.get("confirmed", {}).get("idx", []))
        if confirmed_count > 0:
            pivot_details = []
            for i in range(confirmed_count):
                idx = pm["confirmed"]["idx"][i]
                y = pm["confirmed"]["y"][i]
                ptype = pm["confirmed"]["type"][i]
                # 인덱스를 시간으로 변환
                if 0 <= idx < len(df):
                    ts = df.index[idx]
                    time_str = str(ts).split(" ")[1][:5] if len(str(ts).split(" ")) > 1 else "?"
                else:
                    time_str = "?"
                pivot_details.append(f"{ptype}@{time_str}={y:.2f}")
            logger.info("[ChartEngine][PM] 빌드된 확정 피봇: %s", ", ".join(pivot_details))
        
        logger.info(
            "[ChartEngine][RT] pivot_markers 완성: confirmed=%d unconfirmed=%d elapsed=%.3fs",
            confirmed_count,
            len(pm.get("unconfirmed",{}).get("idx", [])),
            _pm_elapsed,
        )
        return pm

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def compute(
        self,
        df_raw: pd.DataFrame,
        cfg: Optional[Any] = None,
        data_source: Optional[str] = None,
        force_recompute: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """
        분봉 DataFrame → 렌더용 df + pivot_markers.

        Args:
            force_recompute: 범위 변경 시 강제 재계산 (pm 재계산용)

        Returns
        -------
        df_render      : Open/High/Low/Close/Volume, DatetimeIndex (KST naive)
        pivot_markers  : SkyPlot 호환 딕셔너리 | None
        """
        if df_raw is None or df_raw.empty:
            logger.warning("[ChartEngine][RT] compute: df_raw None 또는 비어있음 — 데이터 미수신")
            return pd.DataFrame(), None

        import time as _time
        _rt_t0 = _time.perf_counter()
        logger.info(
            "[ChartEngine][RT] compute 시작: bars=%d last=%s ds=%s",
            len(df_raw),
            df_raw.index[-1] if len(df_raw) else "N/A",
            data_source,
        )
        df = df_raw.copy()

        # 강제 재계산 시 캐시 초기화
        if force_recompute:
            self._last_sig = None
            self._replay_signature = None
            self._zz_state_cache = {}
            self._confirmed_pivots_cache = []
            self._anchor_ts = None
            self._st_state_cache = {}  # SuperTrend 캐시 초기화
            self._st_cache_sig = None  # SuperTrend 캐시 서명 초기화
            self._st_cache_values = None  # SuperTrend 값 캐시 초기화
            self._st_cache_dirs = None  # SuperTrend 방향 캐시 초기화
            self._st_fed_bars = 0  # SuperTrend feed한 완결봉 수 초기화
            if self._st is not None:
                self._st.reset()  # SuperTrend 인스턴스 리셋
            logger.debug("[ChartEngine] 강제 재계산 - 캐시 전체 무효화")

        # 데이터 소스 업데이트
        if data_source is not None:
            # 데이터 소스 변경 시 모든 캐시 초기화 및 ZigZag 재초기화
            if self._current_data_source is not None and self._current_data_source != data_source:
                # [FIX] 외부 주입 인스턴스가 있으면 _zz를 유지
                zz_external = getattr(self, '_zz_external', False)
                zz_before = self._zz
                
                self._last_completed_ts = None
                self._anchor_ts = None
                self._last_sig = None          # [BUG-FIX] 누락됐던 _last_sig 초기화
                self._zz_state_cache = {}  # ZigZag 상태 캐시 초기화
                self._replay_signature = None  # replay 서명 초기화
                self._confirmed_pivots_cache = []   # [FIX] None → [] 타입 통일
                # SuperTrend 캐시 초기화 (전체 재계산 강제)
                self._st_cache_sig = None  # SuperTrend 캐시 서명 초기화
                self._st_cache_values = None  # SuperTrend 값 캐시 초기화
                self._st_cache_dirs = None  # SuperTrend 방향 캐시 초기화
                self._st_fed_bars = 0  # SuperTrend feed한 완결봉 수 초기화
                
                # [FIX] 캐시 초기화 후 _zz 복원 (외부 인스턴스 유지)
                if zz_external and zz_before is not None:
                    self._zz = zz_before
                    logger.info("[ChartEngine] 캐시 초기화 후 _zz 복원 완료: _zz=%s", self._zz is not None)
                
                # 데이터 소스 변경 시 ZigZag 재초기화하여 _zz_cfg 업데이트
                # [FIX] 외부 주입 인스턴스가 있으면 재초기화하지 않음 (Predictor ZigZag 인스턴스 유지)
                logger.info("[ChartEngine] 데이터 소스 변경 감지: %s → %s, _zz_external=%s, _zz=%s", 
                            self._current_data_source, data_source, zz_external, self._zz is not None)
                if not zz_external:
                    logger.debug("[ChartEngine] 데이터 소스 변경 - 캐시 초기화 및 ZigZag 재초기화: %s → %s", self._current_data_source, data_source)
                    self._init_zigzag(cfg, data_source)
                else:
                    logger.info("[ChartEngine] 데이터 소스 변경 - 외부 ZigZag 인스턴스 유지 (재초기화 스킵): %s → %s", self._current_data_source, data_source)
                    # 외부 인스턴스는 reset만 수행하여 상태 초기화
                    if self._zz is not None and hasattr(self._zz, 'reset'):
                        try:
                            self._zz.reset()
                            logger.info("[ChartEngine] 외부 ZigZag reset 완료")
                        except Exception as e:
                            logger.warning("[ChartEngine] 외부 ZigZag reset 실패: %s", e)
                    else:
                        logger.warning("[ChartEngine] 외부 ZigZag 인스턴스가 None이거나 reset 메서드 없음")
            self._current_data_source = data_source

        # 컬럼명 정규화 (tick_processor → lowercase, 필요시 capitalize)
        col_map = {c: c.capitalize()
                   for c in df.columns
                   if c in ("open", "high", "low", "close", "volume")}
        df = df.rename(columns=col_map)

        # timestamp 컬럼 → DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                try:
                    df = df.set_index("timestamp")
                except Exception:
                    pass
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass

        # tz 제거 (KST naive)
        try:
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_convert("Asia/Seoul").tz_localize(None)
        except Exception:
            pass

        # OHLC numeric 강제 변환
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        _nan_rows = int(df[["Open","High","Low","Close"]].isnull().any(axis=1).sum()) if all(c in df.columns for c in ["Open","High","Low","Close"]) else 0
        if _nan_rows > 0:
            logger.warning("[ChartEngine][RT] OHLC NaN 봉 %d개 감지 — 해당 봉 ZigZag 스킵됨", _nan_rows)

        # ── [FIX-4] max_bars 슬라이싱: 앵커 타임스탬프 기준으로 고정 ─────────────
        if len(df) > self._max_bars:
            # 단순 tail() 대신 앵커 타임스탬프 기반 슬라이싱
            # 앵커: 이전 슬라이싱과 동일한 시작점 유지
            anchor_ts = getattr(self, "_anchor_ts", None)
            if anchor_ts is not None and anchor_ts in df.index:
                # 이전 앵커 이후 데이터만 사용
                df = df[df.index >= anchor_ts].copy()
                # 여전히 max_bars 초과 시 tail() 적용 후 앵커 갱신
                if len(df) > self._max_bars:
                    df = df.tail(self._max_bars).copy()
                    self._anchor_ts = df.index[0]
            else:
                df = df.tail(self._max_bars).copy()
                self._anchor_ts = df.index[0]  # 앵커 초기화
        # ────────────────────────────────────────────────────────────────────────

        # ZigZag lazy init
        if self._zz is None:
            # cfg가 None이면 기본 config 로드 시도
            if cfg is None:
                try:
                    from config import AppConfig
                    cfg = AppConfig.from_file("config.json")
                    logger.info("[ChartEngine] cfg가 None이어서 AppConfig.from_file()로 다시 로드")
                except Exception as e:
                    logger.warning("[ChartEngine] AppConfig 로드 실패: %s", e)
            self._init_zigzag(cfg, data_source)

        # SuperTrend lazy init
        if self._st is None:
            # cfg가 None이면 기본 config 로드 시도
            if cfg is None:
                try:
                    from config import AppConfig
                    cfg = AppConfig.from_file("config.json")
                    logger.info("[ChartEngine] cfg가 None이어서 AppConfig.from_file()로 다시 로드")
                except Exception as e:
                    logger.warning("[ChartEngine] AppConfig 로드 실패: %s", e)
            self._init_supertrend(cfg)

        # ── [FIX-2] 캐시 키: 길이 제거, 마지막 완결봉 타임스탬프 기준 ────────────
        # 마지막 봉은 미완결이므로 캐시 키에서 제외
        # 완결봉 기준 = 현재 분봉 시작 시각 이전의 마지막 봉
        # id(self._zz) 제거: 캐시 교체 시 불안정 방지
        try:
            now_floor = pd.Timestamp.now().floor("1min")
            completed_df = df[df.index < now_floor]
            if completed_df.empty:
                completed_df = df.iloc[:-1]

            sig: Optional[tuple] = (
                str(completed_df.index[-1]),          # 마지막 완결봉 시각 (고정)
                len(completed_df),                     # 완결봉 수
                round(float(completed_df["Close"].iloc[-1]), 2),  # 완결봉 종가
            )
        except Exception:
            sig = None
        # ────────────────────────────────────────────────────────────────────────

        if sig is not None and sig == self._last_sig:
            logger.debug("[ChartEngine] compute: 캐시 히트 (sig=%s)", sig)
            pm = self._build_pivot_markers(df)
            # ── SuperTrend: 캐시 히트 경로 ───────────────────────────────────────
            # [SSOT] 외부 주입 인스턴스: _apply_supertrend_from_instance 위임
            # 자체 관리 인스턴스: 완결봉 캐시 재사용 + 미완결봉 1봉 증분
            if self._st is not None:
                try:
                    if self._st_external:
                        # [SSOT] 외부 주입 — 인스턴스에서 직접 df 컬럼 추출
                        df = self._apply_supertrend_from_instance(df, self._st)
                        logger.debug("[ChartEngine][SSOT] 캐시히트 ST 외부주입 적용")
                    elif (self._st_cache_sig == sig
                            and self._st_cache_values is not None
                            and len(self._st_cache_values) > 0):
                        # 완결봉 캐시 재사용 + 미완결봉 1봉 계산
                        last_row   = df.iloc[-1]
                        live_state = self._st.update(
                            float(last_row["High"]),
                            float(last_row["Low"]),
                            float(last_row["Close"]),
                        )
                        st_all   = list(self._st_cache_values) + [live_state.value]
                        st_d_all = list(self._st_cache_dirs)   + [live_state.direction]
                        n_df = len(df)
                        if len(st_all) == n_df:
                            df["SuperTrend"]     = st_all
                            df["SuperTrend_Dir"] = st_d_all
                        elif len(st_all) < n_df:
                            pad = n_df - len(st_all)
                            fv      = next((v for v in st_all   if not np.isnan(v)), np.nan)
                            fv_dir  = next((v for v in st_d_all if not np.isnan(v)), np.nan)
                            df["SuperTrend"]     = [fv]     * pad + st_all
                            df["SuperTrend_Dir"] = [fv_dir] * pad + st_d_all
                            logger.debug(
                                "[ChartEngine] ST 캐시 패딩: cache=%d df=%d pad=%d",
                                len(st_all), n_df, pad,
                            )
                        else:
                            trimmed   = st_all[-n_df:]
                            trimmed_d = st_d_all[-n_df:]
                            if np.all(np.isnan(trimmed)):
                                logger.warning("[ChartEngine] ST 트림 후 모든 NaN - 캐시 초기화")
                                self._st_cache_sig    = None
                                self._st_cache_values = None
                                self._st_cache_dirs   = None
                                self._st_fed_bars     = 0
                            else:
                                df["SuperTrend"]     = trimmed
                                df["SuperTrend_Dir"] = trimmed_d
                        logger.debug(
                            "[ChartEngine] ST 캐시 히트+증분: 완결=%d봉",
                            len(self._st_cache_values),
                        )
                    else:
                        # 캐시 없음 → 캐시 미스 경로에서 처리
                        pass
                except Exception as e:
                    logger.warning("[ChartEngine] ST 캐시 히트 증분 실패: %s", e)
            # ────────────────────────────────────────────────────────────────────
            return df, pm

        logger.info("[ChartEngine] compute: 새 계산 필요 (sig=%s, force_recompute=%s, _last_sig=%s)", sig, force_recompute, self._last_sig)

        # ── SuperTrend 계산 (캐시 미스 경로) ────────────────────────────────────
        # [SSOT] 외부 주입(_st_external=True): _apply_supertrend_from_instance
        # 자체 관리(_st_external=False):       _compute_supertrend_replay
        # [SUPERTREND-INTEGRATION] ZigZag보다 먼저 계산하여 SuperTrend_Dir 컬럼 제공
        if self._st is not None:
            if self._st_external:
                df = self._apply_supertrend_from_instance(df, self._st)
                logger.debug("[ChartEngine][SSOT] 캐시미스 ST 외부주입 적용")
            else:
                df = self._compute_supertrend_replay(df)
                # _compute_supertrend_replay가 캐시 서명을 내부적으로 저장하므로
                # compute() 레벨에서 sig 재할당
                self._st_cache_sig = sig
        # ────────────────────────────────────────────────────────────────────────

        # 실시간 모드에서 미완결 봉을 제외하여 피봇 노이즈 방지
        exclude_last = True
        self._feed_zigzag(df, exclude_last=exclude_last, force_recompute=force_recompute)
        logger.debug("[ChartEngine] compute: _feed_zigzag 완료")
        logger.debug("[ChartEngine] compute: _zz 상태 확인: _zz=%s", self._zz is not None)
        pm = self._build_pivot_markers(df)
        self._last_sig = sig

        _rt_elapsed = _time.perf_counter() - _rt_t0
        _pm_conf  = len(pm.get("confirmed",  {}).get("idx", [])) if pm else 0
        _pm_unconf= len(pm.get("unconfirmed",{}).get("idx", [])) if pm else 0
        logger.info(
            "[ChartEngine][RT] compute 완료: bars=%d pivot(확정=%d 미확정=%d) elapsed=%.3fs",
            len(df), _pm_conf, _pm_unconf, _rt_elapsed,
        )
        return df, pm


def _nearest_bar_idx(
    ts_index: pd.DatetimeIndex,
    target:   pd.Timestamp,
    tol_sec:  int = 90,
) -> int:
    """타임스탬프 근접 매칭. tol_sec 초 이내 없으면 -1."""
    try:
        # pd.DatetimeIndex → numpy datetime64[ns] 로 통일한 뒤 int64(ns) 비교
        arr  = ts_index.astype("datetime64[ns]").astype("int64")  # ns 단위 int64
        t0   = int(pd.Timestamp(target).value)                    # ns 단위 int64
        diff = np.abs(arr - t0)
        i    = int(np.argmin(diff))
        if diff[i] <= tol_sec * 1_000_000_000:
            return i
    except Exception:
        pass
    return -1
