"""선물-콜 유사도 분석 (FuturesCallSimilarity).

DTW 거리, Rolling correlation, Beta/R² 등을 이용해
선물 가격과 ATM 콜 옵션 가격의 구조적 유사도를 측정한다.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import math
import numpy as np

class FuturesCallSimilarity:
    """KP200 선물가와 ATM 콜 가격의 추적 유사도를 수치화한다.

    만기가 근접하면 감마 지배 구간에 돌입하여 ATM 콜이 선물을 선형 추적하지 않게 된다.
    이 클래스는 세 가지 지표를 복합한 CDS(Composite Divergence Score)로 이탈 정도를
    0~1 범위로 출력하여, 방향성 콜 매매 위험을 사전 경보한다.

    Attributes:
        window (int): 롤링 윈도우 크기 (기본 20틱).

    CDS 구성:
        corr_term  = 1 - Pearson(delta_fut, delta_call/delta)   [0,1]
        dtw_term   = DTW_normalized(z_fut, z_call) / window     [0,∞ → 클리핑]
        r2_term    = 1 - R²(ΔCall ~ β·ΔFut)                    [0,1]

        CDS = w_corr·corr_term + w_dtw·dtw_term + w_r2·r2_term  ∈ [0,1]

    임계값 가이드:
        CDS < 0.3  : 선물 추적 양호 — 방향성 매매 가능
        CDS 0.3~0.5: 주의 구간 — 감마 영향 증가
        CDS > 0.5  : 경보 — 감마 지배, 방향성 콜 매매 위험
        r2 < 0.5   : 감마 지배 확정 (통상 D-3 이내)
    """

    def __init__(self, window: int = 20) -> None:
        self.window = max(int(window), 5)

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _zscore(arr: "np.ndarray") -> "np.ndarray":
        """z-score 정규화. std=0 이면 zeros 반환."""
        try:
            mu = np.mean(arr)
            sd = np.std(arr)
            if sd < 1e-12:
                return np.zeros_like(arr)
            return (arr - mu) / sd
        except Exception:
            return np.zeros_like(arr)

    # ── 공개 메서드 ────────────────────────────────────────────────────────────

    def rolling_correlation(
        self,
        fut_prices: "np.ndarray",
        call_prices: "np.ndarray",
        delta: float = 0.5,
    ) -> float:
        """델타 조정 수익률 기반 롤링 Pearson 상관계수를 반환한다.

        Args:
            fut_prices:  선물 체결가 배열 (길이 >= 2, 오름차순 시간).
            call_prices: ATM 콜 체결가 배열 (fut_prices와 동일 길이).
            delta:       ATM 콜 델타 (이론값 0.5; 실시간 delta 필드가 있으면 전달).

        Returns:
            Pearson 상관계수 [-1, 1]. 계산 불가 시 0.0.
        """
        try:
            f = np.asarray(fut_prices, dtype=np.float64)
            c = np.asarray(call_prices, dtype=np.float64)
            n = min(len(f), len(c), self.window)
            if n < 2:
                return 0.0
            f, c = f[-n:], c[-n:]

            d_f = np.diff(f)
            d_c = np.diff(c)
            delta_f = float(max(abs(delta), 1e-6))
            d_c_adj = d_c / delta_f  # 델타 정규화: 콜 이동 ÷ delta

            mask = np.isfinite(d_f) & np.isfinite(d_c_adj)
            if mask.sum() < 2:
                return 0.0

            from scipy.stats import pearsonr
            corr, _ = pearsonr(d_f[mask], d_c_adj[mask])
            return float(np.clip(corr if np.isfinite(corr) else 0.0, -1.0, 1.0))
        except Exception:
            return 0.0

    def dtw_distance(
        self,
        fut_prices: "np.ndarray",
        call_prices: "np.ndarray",
    ) -> float:
        """z-score 정규화 후 DTW 거리를 window로 나눈 정규화 값을 반환한다.

        fastdtw 패키지 필요: pip install fastdtw

        Returns:
            DTW_normalized ≥ 0. 0에 가까울수록 형태 유사. 계산 불가 시 0.0.
        """
        try:
            from fastdtw import fastdtw  # type: ignore
            from scipy.spatial.distance import euclidean

            f = np.asarray(fut_prices, dtype=np.float64)
            c = np.asarray(call_prices, dtype=np.float64)
            n = min(len(f), len(c), self.window)
            if n < 2:
                return 0.0
            f, c = f[-n:], c[-n:]

            zf = self._zscore(f)
            zc = self._zscore(c)

            dist, _ = fastdtw(zf.reshape(-1, 1), zc.reshape(-1, 1), dist=euclidean)
            result = float(dist) / float(n)
            return float(result) if np.isfinite(result) else 0.0
        except ImportError:
            # fastdtw 없을 때: 단순 유클리드 거리 fallback
            try:
                f = np.asarray(fut_prices, dtype=np.float64)
                c = np.asarray(call_prices, dtype=np.float64)
                n = min(len(f), len(c), self.window)
                if n < 2:
                    return 0.0
                zf = self._zscore(f[-n:])
                zc = self._zscore(c[-n:])
                dist = float(np.sqrt(np.mean((zf - zc) ** 2)))
                return float(dist) if np.isfinite(dist) else 0.0
            except Exception:
                return 0.0
        except Exception:
            return 0.0

    def beta_r2(
        self,
        fut_prices: "np.ndarray",
        call_prices: "np.ndarray",
    ) -> "tuple[float, float]":
        """OLS 회귀 ΔCall ~ β·ΔFut 의 (beta, R²) 를 반환한다.

        R² 저하: 감마 지배 구간 진입 신호.
            R² ≈ 1.0  : 콜이 선물을 선형 추적 중
            R² < 0.5  : 감마 비선형성 지배 → 방향성 콜 위험

        Returns:
            (beta, r2) tuple. 계산 불가 시 (0.5, 0.0).
        """
        try:
            f = np.asarray(fut_prices, dtype=np.float64)
            c = np.asarray(call_prices, dtype=np.float64)
            n = min(len(f), len(c), self.window)
            if n < 3:
                return (0.5, 0.0)
            f, c = f[-n:], c[-n:]

            d_f = np.diff(f)
            d_c = np.diff(c)
            mask = np.isfinite(d_f) & np.isfinite(d_c)
            if mask.sum() < 3:
                return (0.5, 0.0)

            x = d_f[mask]
            y = d_c[mask]

            # OLS: β = Cov(x,y)/Var(x)
            var_x = float(np.var(x))
            if var_x < 1e-12:
                return (0.5, 0.0)
            beta = float(np.cov(x, y)[0, 1] / var_x)

            # R²: 1 - SS_res/SS_tot
            y_hat = beta * x
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
            r2 = float(np.clip(r2, 0.0, 1.0))

            return (
                float(beta) if np.isfinite(beta) else 0.5,
                float(r2)   if np.isfinite(r2)   else 0.0,
            )
        except Exception:
            return (0.5, 0.0)

    def composite_divergence_score(
        self,
        fut_prices: "np.ndarray",
        call_prices: "np.ndarray",
        delta: float = 0.5,
        *,
        w_corr: float = 0.4,
        w_dtw:  float = 0.3,
        w_r2:   float = 0.3,
    ) -> "Dict[str, float]":
        """CDS(Composite Divergence Score)와 구성 지표를 계산하여 반환한다.

        Args:
            fut_prices:  선물 체결가 배열.
            call_prices: ATM 콜 체결가 배열.
            delta:       ATM 콜 델타 (기본 0.5).
            w_corr:      상관계수 항 가중치 (기본 0.4).
            w_dtw:       DTW 항 가중치 (기본 0.3).
            w_r2:        R² 항 가중치 (기본 0.3).

        Returns:
            {
                "cds":  float,  # Composite Divergence Score [0,1]
                "corr": float,  # Pearson 상관계수 [-1,1]
                "dtw":  float,  # DTW 정규화 거리 [0,∞)
                "r2":   float,  # OLS R² [0,1]
                "beta": float,  # OLS 베타 계수
                "n_samples": int,  # 사용된 샘플 수
            }
        """
        empty: Dict[str, float] = {"cds": 0.0, "corr": 0.0, "dtw": 0.0, "r2": 0.0, "beta": 0.5, "n_samples": 0}

        try:
            f = np.asarray(fut_prices, dtype=np.float64)
            c = np.asarray(call_prices, dtype=np.float64)
            n = min(len(f), len(c), self.window)
            if n < 3:
                return empty

            corr = self.rolling_correlation(f, c, delta)
            dtw  = self.dtw_distance(f, c)
            beta, r2 = self.beta_r2(f, c)

            # 각 항 [0,1] 변환
            corr_term = float(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))  # corr=1→0, corr=-1→1
            dtw_term  = float(np.clip(dtw, 0.0, 1.0))                  # dtw는 이미 window 정규화
            r2_term   = float(np.clip(1.0 - r2, 0.0, 1.0))            # r2=1→0, r2=0→1

            cds = (
                float(w_corr) * corr_term
                + float(w_dtw) * dtw_term
                + float(w_r2)  * r2_term
            )
            cds = float(np.clip(cds, 0.0, 1.0))
            if not np.isfinite(cds):
                cds = 0.0

            return {
                "cds":       round(cds, 4),
                "corr":      round(corr, 4),
                "dtw":       round(dtw, 4),
                "r2":        round(r2, 4),
                "beta":      round(beta, 4),
                "n_samples": int(n),
            }
        except Exception:
            return empty

    def dte_profile(
        self,
        records: "list[dict]",
    ) -> "Dict[str, float]":
        """DTE 구간별 평균 CDS를 계산하여 만기 근접에 따른 이탈 패턴을 분석한다.

        Args:
            records: [{"dte": float, "cds": float}, ...] 리스트.
                     pipeline의 히스토리 큐에서 수집한 값을 전달.

        Returns:
            {"dte_0_1": float, "dte_2_5": float, "dte_6_10": float, "dte_11p": float}
            각 구간의 평균 CDS. 데이터 없는 구간은 0.0.
        """
        buckets: Dict[str, list] = {
            "dte_0_1":  [],
            "dte_2_5":  [],
            "dte_6_10": [],
            "dte_11p":  [],
        }
        try:
            for r in (records or []):
                dte = float(r.get("dte") or 0.0)
                cds = float(r.get("cds") or 0.0)
                if not (np.isfinite(dte) and np.isfinite(cds)):
                    continue
                if dte <= 1.0:
                    buckets["dte_0_1"].append(cds)
                elif dte <= 5.0:
                    buckets["dte_2_5"].append(cds)
                elif dte <= 10.0:
                    buckets["dte_6_10"].append(cds)
                else:
                    buckets["dte_11p"].append(cds)
        except Exception:
            pass

        return {
            k: round(float(np.mean(v)), 4) if v else 0.0
            for k, v in buckets.items()
        }
