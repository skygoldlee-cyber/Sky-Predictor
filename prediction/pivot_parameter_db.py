"""Pivot Parameter Database

피봇 탐지 파라미터를 일일 단위로 저장하고 분석하여
최적 파라미터를 학습하는 데이터베이스.

Usage:
    from prediction.pivot_parameter_db import PivotParameterDB
    
    db = PivotParameterDB("pivot_parameters.db")
    
    # 일일 파라미터 저장
    db.save_daily_parameters(
        date="2026-05-19",
        symbol="KP200 선물",
        indicator_type="adaptive_zigzag",
        config={
            "atr_multiplier": 1.5,
            "atr_period": 14,
            "base_pct": 0.3,
            "atr_weight": 0.5,
            ...
        },
        performance_metrics={
            "total_pivots": 10,
            "confirmed_pivots": 8,
            "pivot_confirmation_rate": 0.8,
            ...
        },
        market_state={
            "market_structure": "uptrend",
            "avg_atr": 2.5,
            ...
        }
    )
    
    # 최적 파라미터 조회
    best_params = db.query_best_parameters(
        symbol="KP200 선물",
        market_structure="uptrend",
        lookback_days=30
    )
"""

import logging
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


class PivotParameterDB:
    """피봇 파라미터 데이터베이스."""
    
    def __init__(self, db_path: str = "data/pivot_parameters.db"):
        """초기화.
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        # 디렉토리 생성
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # WAL 모드: 읽기-쓰기 동시 접근 허용
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()
        _logger.info("[PIVOT_PARAM_DB] 데이터베이스 초기화: %s", db_path)
    
    def _create_tables(self):
        """테이블 생성."""
        cursor = self.conn.cursor()
        
        # pivot_parameters_daily 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pivot_parameters_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                indicator_type TEXT NOT NULL,
                
                -- ATR 관련 파라미터
                atr_multiplier REAL,
                atr_period INTEGER,
                atr_multiplier_min REAL,
                atr_multiplier_max REAL,
                
                -- 퍼센트 관련 파라미터
                base_pct REAL,
                min_wave_pct REAL,
                
                -- 하이브리드 관련 파라미터
                atr_weight REAL,
                
                -- 공통 파라미터
                er_period INTEGER,
                confirmation_bars INTEGER,
                min_wave_bars INTEGER,
                major_swing_ratio REAL,
                
                -- 성능 메트릭 (학습 자료)
                total_pivots INTEGER,
                confirmed_pivots INTEGER,
                cancelled_pivots INTEGER,
                pivot_confirmation_rate REAL,
                avg_pivot_lifespan_bars REAL,
                avg_wave_size_pct REAL,
                avg_wave_atr_ratio REAL,
                
                -- 시장 상태
                market_structure TEXT,
                avg_atr REAL,
                price_volatility REAL,
                
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # pivot_parameters_session 테이블 (세션·레짐 단위)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pivot_parameters_session (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT NOT NULL,
                session_label   TEXT NOT NULL,
                time_start      TEXT,
                time_end        TEXT,
                bar_count       INTEGER,
                symbol          TEXT NOT NULL,
                indicator_type  TEXT NOT NULL,

                -- 파라미터 (버킷팅 적용)
                atr_multiplier_bin  REAL,
                base_pct_bin        REAL,
                atr_weight_bin      REAL,
                base_multiplier_bin REAL,
                confirmation_bars   INTEGER,
                er_period           INTEGER,
                min_wave_pct_bin    REAL,

                -- 레짐 컨텍스트
                dominant_regime     TEXT,
                regime_stability    REAL,
                avg_atr             REAL,
                avg_er              REAL,
                atr_percentile      REAL,

                -- 다차원 성능 지표
                total_pivots            INTEGER,
                confirmed_pivots        INTEGER,
                cancelled_pivots        INTEGER,
                pivot_confirmation_rate REAL,
                avg_lag_bars            REAL,
                lag_p95_bars            REAL,
                pivot_quality_score     REAL,
                alternation_rate        REAL,
                avg_wave_size_pct       REAL,
                avg_wave_atr_ratio      REAL,
                false_pivot_rate        REAL,
                composite_score        REAL,

                created_at  TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 복합 인덱스 (쿼리 패턴 최적화)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_lookup ON pivot_parameters_session(symbol, dominant_regime, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_perf ON pivot_parameters_session(pivot_confirmation_rate, avg_lag_bars)")
        
        # 인덱스 생성
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pivot_params_date ON pivot_parameters_daily(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pivot_params_symbol ON pivot_parameters_daily(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pivot_params_indicator ON pivot_parameters_daily(indicator_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pivot_params_structure ON pivot_parameters_daily(market_structure)")
        
        self.conn.commit()
    
    def save_daily_parameters(
        self,
        date: str,
        symbol: str,
        indicator_type: str,
        config: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        market_state: Dict[str, Any]
    ) -> None:
        """일일 파라미터 저장.
        
        Args:
            date: 날짜 (YYYY-MM-DD)
            symbol: 심볼명 (KP200 선물, KOSPI 지수)
            indicator_type: 지표 타입 (adaptive_zigzag, hybrid_adaptive_pivot)
            config: 파라미터 설정 딕셔너리
            performance_metrics: 성능 메트릭 딕셔너리
            market_state: 시장 상태 딕셔너리
        """
        cursor = self.conn.cursor()
        
        # 파라미터 추출 (기본값 None)
        # atr_multiplier와 base_multiplier 호환성 처리
        atr_multiplier = config.get("atr_multiplier") or config.get("base_multiplier")
        atr_period = config.get("atr_period")
        atr_multiplier_min = config.get("atr_multiplier_min")
        atr_multiplier_max = config.get("atr_multiplier_max")
        base_pct = config.get("base_pct")
        min_wave_pct = config.get("min_wave_pct")
        atr_weight = config.get("atr_weight")
        er_period = config.get("er_period")
        confirmation_bars = config.get("confirmation_bars")
        min_wave_bars = config.get("min_wave_bars")
        major_swing_ratio = config.get("major_swing_ratio")
        
        # 성능 메트릭 추출
        total_pivots = performance_metrics.get("total_pivots")
        confirmed_pivots = performance_metrics.get("confirmed_pivots")
        cancelled_pivots = performance_metrics.get("cancelled_pivots")
        pivot_confirmation_rate = performance_metrics.get("pivot_confirmation_rate")
        avg_pivot_lifespan_bars = performance_metrics.get("avg_pivot_lifespan_bars")
        avg_wave_size_pct = performance_metrics.get("avg_wave_size_pct")
        avg_wave_atr_ratio = performance_metrics.get("avg_wave_atr_ratio")
        
        # 시장 상태 추출
        market_structure = market_state.get("market_structure")
        avg_atr = market_state.get("avg_atr")
        price_volatility = market_state.get("price_volatility")
        
        cursor.execute("""
            INSERT INTO pivot_parameters_daily (
                date, symbol, indicator_type,
                atr_multiplier, atr_period, atr_multiplier_min, atr_multiplier_max,
                base_pct, min_wave_pct, atr_weight,
                er_period, confirmation_bars, min_wave_bars, major_swing_ratio,
                total_pivots, confirmed_pivots, cancelled_pivots,
                pivot_confirmation_rate, avg_pivot_lifespan_bars,
                avg_wave_size_pct, avg_wave_atr_ratio,
                market_structure, avg_atr, price_volatility
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date, symbol, indicator_type,
            atr_multiplier, atr_period, atr_multiplier_min, atr_multiplier_max,
            base_pct, min_wave_pct, atr_weight,
            er_period, confirmation_bars, min_wave_bars, major_swing_ratio,
            total_pivots, confirmed_pivots, cancelled_pivots,
            pivot_confirmation_rate, avg_pivot_lifespan_bars,
            avg_wave_size_pct, avg_wave_atr_ratio,
            market_structure, avg_atr, price_volatility
        ))
        
        self.conn.commit()
        _logger.info(
            "[PIVOT_PARAM_DB] 일일 파라미터 저장: %s %s %s",
            date, symbol, indicator_type
        )
    
    def query_best_parameters(
        self,
        symbol: str,
        market_structure: Optional[str] = None,
        indicator_type: Optional[str] = None,
        lookback_days: int = 30,
        min_pivots: int = 5
    ) -> Dict[str, Any]:
        """시장 상태별 최적 파라미터 조회.
        
        Args:
            symbol: 심볼명
            market_structure: 시장 구조 (uptrend/downtrend/ranging/unknown), None이면 전체
            indicator_type: 지표 타입, None이면 전체
            lookback_days: 조회 기간 (일)
            min_pivots: 최소 피봇 수 (이하인 데이터는 제외)
        
        Returns:
            최적 파라미터 딕셔너리
        """
        cursor = self.conn.cursor()
        
        # 날짜 범위 계산
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        query = """
            SELECT 
                atr_multiplier, atr_period, atr_multiplier_min, atr_multiplier_max,
                base_pct, min_wave_pct, atr_weight,
                er_period, confirmation_bars, min_wave_bars, major_swing_ratio,
                AVG(pivot_confirmation_rate) as avg_confirmation_rate,
                AVG(avg_pivot_lifespan_bars) as avg_lifespan,
                COUNT(*) as sample_count
            FROM pivot_parameters_daily
            WHERE symbol = ?
                AND date >= ?
                AND total_pivots >= ?
        """
        params = [symbol, start_date.strftime("%Y-%m-%d"), min_pivots]
        
        if market_structure:
            query += " AND market_structure = ?"
            params.append(market_structure)
        
        if indicator_type:
            query += " AND indicator_type = ?"
            params.append(indicator_type)
        
        query += " GROUP BY ROUND(atr_multiplier, 1), ROUND(base_pct, 2), ROUND(atr_weight, 1) ORDER BY avg_confirmation_rate DESC LIMIT 1"
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if not row:
            _logger.warning("[PIVOT_PARAM_DB] 최적 파라미터 조회 실패: 조건에 맞는 데이터 없음")
            return {}
        
        columns = [
            "atr_multiplier", "atr_period", "atr_multiplier_min", "atr_multiplier_max",
            "base_pct", "min_wave_pct", "atr_weight",
            "er_period", "confirmation_bars", "min_wave_bars", "major_swing_ratio",
            "avg_confirmation_rate", "avg_lifespan", "sample_count"
        ]
        
        result = dict(zip(columns, row))
        _logger.info(
            "[PIVOT_PARAM_DB] 최적 파라미터 조회: %s %s 확정률=%.2f 샘플=%d",
            symbol, market_structure or "전체", result["avg_confirmation_rate"], result["sample_count"]
        )
        
        return result
    
    def query_best_parameters_session(
        self,
        symbol: str,
        dominant_regime: Optional[str] = None,
        indicator_type: Optional[str] = None,
        lookback_days: int = 30,
        min_pivots: int = 5,
        min_sample: int = 3,
    ) -> Dict[str, Any]:
        """세션·레짐 단위 최적 파라미터 조회 (복합 점수 기반).
        
        Args:
            symbol: 심볼명
            dominant_regime: 레짐 (trend_strong/chop/volatile/mixed), None이면 전체
            indicator_type: 지표 타입, None이면 전체
            lookback_days: 조회 기간 (일)
            min_pivots: 최소 피봇 수
            min_sample: 최소 샘플 수
        
        Returns:
            최적 파라미터 딕셔너리
        """
        cursor = self.conn.cursor()
        
        # 날짜 범위 계산
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        query = """
            SELECT
                atr_multiplier_bin,
                base_pct_bin,
                atr_weight_bin,
                base_multiplier_bin,
                confirmation_bars,
                er_period,
                min_wave_pct_bin,
                COUNT(*)                            AS sample_count,
                AVG(pivot_confirmation_rate)        AS avg_confirmation_rate,
                AVG(avg_lag_bars)                   AS avg_lag,
                AVG(pivot_quality_score)            AS avg_quality,
                AVG(alternation_rate)               AS avg_alternation,
                AVG(false_pivot_rate)               AS avg_false_rate,
                AVG(
                    pivot_confirmation_rate * 0.35
                    + (1.0 - MIN(avg_lag_bars / 20.0, 1.0)) * 0.25
                    + pivot_quality_score * 0.20
                    + alternation_rate * 0.10
                    + (1.0 - false_pivot_rate) * 0.10
                )                                   AS composite_score
            FROM pivot_parameters_session
            WHERE symbol = ?
              AND date >= ?
              AND total_pivots >= ?
        """
        params = [symbol, start_date.strftime("%Y-%m-%d"), min_pivots]
        
        if dominant_regime:
            query += " AND dominant_regime = ?"
            params.append(dominant_regime)
        
        if indicator_type:
            query += " AND indicator_type = ?"
            params.append(indicator_type)
        
        query += """
            GROUP BY
                atr_multiplier_bin,
                base_pct_bin,
                atr_weight_bin,
                base_multiplier_bin,
                confirmation_bars
            HAVING sample_count >= ?
            ORDER BY composite_score DESC
            LIMIT 5
        """
        params.append(min_sample)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            _logger.warning("[PIVOT_PARAM_DB] 세션 파라미터 조회 실패: 조건에 맞는 데이터 없음")
            return {}
        
        columns = [
            "atr_multiplier_bin", "base_pct_bin", "atr_weight_bin", "base_multiplier_bin",
            "confirmation_bars", "er_period", "min_wave_pct_bin",
            "sample_count", "avg_confirmation_rate", "avg_lag", "avg_quality",
            "avg_alternation", "avg_false_rate", "composite_score"
        ]
        
        # 상위 1개 반환
        result = dict(zip(columns, rows[0]))
        _logger.info(
            "[PIVOT_PARAM_DB] 세션 파라미터 조회: %s regime=%s composite=%.3f 샘플=%d",
            symbol, dominant_regime or "전체", result["composite_score"], result["sample_count"]
        )
        
        return result
    
    def analyze_parameter_performance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        indicator_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """파라미터 성능 분석.
        
        Args:
            symbol: 심볼명
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            indicator_type: 지표 타입, None이면 전체
        
        Returns:
            성능 분석 결과
        """
        cursor = self.conn.cursor()
        
        query = """
            SELECT 
                market_structure,
                indicator_type,
                COUNT(*) as total_days,
                AVG(total_pivots) as avg_total_pivots,
                AVG(confirmed_pivots) as avg_confirmed_pivots,
                AVG(pivot_confirmation_rate) as avg_confirmation_rate,
                AVG(avg_pivot_lifespan_bars) as avg_lifespan,
                AVG(avg_wave_size_pct) as avg_wave_size_pct,
                AVG(avg_wave_atr_ratio) as avg_wave_atr_ratio
            FROM pivot_parameters_daily
            WHERE symbol = ?
                AND date >= ?
                AND date <= ?
        """
        params = [symbol, start_date, end_date]
        
        if indicator_type:
            query += " AND indicator_type = ?"
            params.append(indicator_type)
        
        query += " GROUP BY market_structure, indicator_type ORDER BY market_structure, indicator_type"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [
            "market_structure", "indicator_type", "total_days",
            "avg_total_pivots", "avg_confirmed_pivots", "avg_confirmation_rate",
            "avg_lifespan", "avg_wave_size_pct", "avg_wave_atr_ratio"
        ]
        
        results = [dict(zip(columns, row)) for row in rows]
        
        # 전체 평균 계산
        if results:
            total_days = sum(r["total_days"] for r in results)
            overall_avg_confirmation_rate = sum(
                r["avg_confirmation_rate"] * r["total_days"] for r in results
            ) / total_days if total_days > 0 else 0.0
        else:
            overall_avg_confirmation_rate = 0.0
        
        analysis = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "indicator_type": indicator_type,
            "overall_avg_confirmation_rate": overall_avg_confirmation_rate,
            "by_market_structure": results
        }
        
        _logger.info(
            "[PIVOT_PARAM_DB] 파라미터 성능 분석: %s %s~%s 전체확정률=%.2f",
            symbol, start_date, end_date, overall_avg_confirmation_rate
        )
        
        return analysis
    
    def query_parameter_distribution(
        self,
        symbol: str,
        parameter_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """파라미터 분포 조회.
        
        Args:
            symbol: 심볼명
            parameter_name: 파라미터명 (atr_multiplier, base_pct, atr_weight 등)
            start_date: 시작 날짜 (YYYY-MM-DD), None이면 전체
            end_date: 종료 날짜 (YYYY-MM-DD), None이면 전체
        
        Returns:
            파라미터 분포 통계
        """
        cursor = self.conn.cursor()
        
        # 파라미터명 유효성 검사
        valid_params = [
            "atr_multiplier", "atr_period", "atr_multiplier_min", "atr_multiplier_max",
            "base_pct", "min_wave_pct", "atr_weight",
            "er_period", "confirmation_bars", "min_wave_bars", "major_swing_ratio"
        ]
        
        if parameter_name not in valid_params:
            _logger.error("[PIVOT_PARAM_DB] 유효하지 않은 파라미터명: %s", parameter_name)
            return {}
        
        query = f"""
            SELECT 
                {parameter_name},
                COUNT(*) as count,
                AVG(pivot_confirmation_rate) as avg_confirmation_rate
            FROM pivot_parameters_daily
            WHERE symbol = ?
                AND {parameter_name} IS NOT NULL
        """
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += f" GROUP BY {parameter_name} ORDER BY {parameter_name}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [parameter_name, "count", "avg_confirmation_rate"]
        distribution = [dict(zip(columns, row)) for row in rows]
        
        # 통계 계산
        if distribution:
            values = [r[parameter_name] for r in distribution]
            weights = [r["count"] for r in distribution]
            
            weighted_avg = sum(v * w for v, w in zip(values, weights)) / sum(weights)
            min_val = min(values)
            max_val = max(values)
        else:
            weighted_avg = 0.0
            min_val = 0.0
            max_val = 0.0
        
        result = {
            "parameter_name": parameter_name,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "weighted_avg": weighted_avg,
            "min": min_val,
            "max": max_val,
            "distribution": distribution
        }
        
        _logger.info(
            "[PIVOT_PARAM_DB] 파라미터 분포 조회: %s %s 가중평균=%.2f",
            symbol, parameter_name, weighted_avg
        )
        
        return result
    
    def get_all_dates(self, symbol: Optional[str] = None) -> List[str]:
        """저장된 날짜 목록 조회.
        
        Args:
            symbol: 심볼명, None이면 전체
        
        Returns:
            날짜 리스트 (YYYY-MM-DD)
        """
        cursor = self.conn.cursor()
        
        query = "SELECT DISTINCT date FROM pivot_parameters_daily WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY date DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        dates = [row[0] for row in rows]
        return dates
    
    def delete_old_records(self, days_to_keep: int = 365) -> None:
        """오래된 레코드 삭제.
        
        Args:
            days_to_keep: 보유 기간 (일)
        """
        cursor = self.conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cursor.execute(
            "DELETE FROM pivot_parameters_daily WHERE date < ?",
            (cutoff_date.strftime("%Y-%m-%d"),)
        )
        
        deleted_count = cursor.rowcount
        self.conn.commit()
        
        _logger.info(
            "[PIVOT_PARAM_DB] 오래된 레코드 삭제: %d건 (%d일 이전)",
            deleted_count, days_to_keep
        )
    
    def _calc_composite_score(
        self,
        confirmation_rate: float,
        avg_lag_bars: float,
        pivot_quality: float,
        alternation_rate: float,
        false_pivot_rate: float,
    ) -> float:
        """가중 복합 점수 계산 (0~1, 높을수록 좋음)."""
        # lag는 최대 20봉으로 정규화 후 반전
        lag_score = max(0.0, 1.0 - avg_lag_bars / 20.0)
        # false pivot은 반전
        fp_score = 1.0 - min(false_pivot_rate, 1.0)

        return (
            confirmation_rate * 0.35
            + lag_score       * 0.25
            + pivot_quality   * 0.20
            + alternation_rate * 0.10
            + fp_score        * 0.10
        )
    
    def save_session_parameters(
        self,
        date: str,
        session_label: str,
        symbol: str,
        indicator_type: str,
        config: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        market_state: Dict[str, Any],
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
    ) -> None:
        """세션·레짐 단위 파라미터 저장."""
        # 파라미터 버킷팅 (연속형 → 반올림 그룹)
        atr_mult_bin    = round(config.get("atr_multiplier", 0) or 0, 1)
        base_pct_bin    = round(config.get("base_pct", 0) or 0, 2)
        atr_weight_bin  = round(config.get("atr_weight", 0) or 0, 1)
        base_mult_bin   = round(
            config.get("base_multiplier") or config.get("atr_multiplier") or 0, 1
        )
        min_wave_bin    = round(config.get("min_wave_pct", 0) or 0, 2)

        # 복합 점수 계산
        composite = self._calc_composite_score(
            confirmation_rate = performance_metrics.get("pivot_confirmation_rate", 0.0),
            avg_lag_bars      = performance_metrics.get("avg_lag_bars", 10.0),
            pivot_quality     = performance_metrics.get("pivot_quality_score", 0.0),
            alternation_rate  = performance_metrics.get("alternation_rate", 0.0),
            false_pivot_rate  = performance_metrics.get("false_pivot_rate", 0.0),
        )

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO pivot_parameters_session (
                date, session_label, time_start, time_end,
                bar_count, symbol, indicator_type,
                atr_multiplier_bin, base_pct_bin, atr_weight_bin,
                base_multiplier_bin, confirmation_bars, er_period, min_wave_pct_bin,
                dominant_regime, regime_stability, avg_atr, avg_er, atr_percentile,
                total_pivots, confirmed_pivots, cancelled_pivots,
                pivot_confirmation_rate, avg_lag_bars, lag_p95_bars,
                pivot_quality_score, alternation_rate, avg_wave_size_pct,
                avg_wave_atr_ratio, false_pivot_rate, composite_score
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )
        """, (
            date, session_label, time_start, time_end,
            performance_metrics.get("bar_count"),
            symbol, indicator_type,
            atr_mult_bin, base_pct_bin, atr_weight_bin,
            base_mult_bin,
            config.get("confirmation_bars"),
            config.get("er_period"),
            min_wave_bin,
            market_state.get("dominant_regime"),
            market_state.get("regime_stability"),
            market_state.get("avg_atr"),
            market_state.get("avg_er"),
            market_state.get("atr_percentile"),
            performance_metrics.get("total_pivots"),
            performance_metrics.get("confirmed_pivots"),
            performance_metrics.get("cancelled_pivots"),
            performance_metrics.get("pivot_confirmation_rate"),
            performance_metrics.get("avg_lag_bars"),
            performance_metrics.get("lag_p95_bars"),
            performance_metrics.get("pivot_quality_score"),
            performance_metrics.get("alternation_rate"),
            performance_metrics.get("avg_wave_size_pct"),
            performance_metrics.get("avg_wave_atr_ratio"),
            performance_metrics.get("false_pivot_rate"),
            composite,
        ))
        self.conn.commit()
        _logger.info(
            "[PIVOT_PARAM_DB] 세션 파라미터 저장: %s %s regime=%s composite=%.3f",
            date, session_label, market_state.get("dominant_regime"), composite
        )
    
    def close(self):
        """DB 연결 종료."""
        self.conn.close()
        _logger.info("[PIVOT_PARAM_DB] DB 연결 종료")


class ParameterRecommender:
    """DB 조회 결과를 실제 Config 객체로 변환하는 추천 레이어."""

    # 레짐별 하드코딩 폴백 (DB 데이터 부족 시)
    REGIME_FALLBACK = {
        "trend_strong_up": {
            "atr_multiplier": 1.8, "base_pct": 0.25,
            "atr_weight": 0.75, "confirmation_bars": 1,
        },
        "trend_strong_dn": {
            "atr_multiplier": 1.8, "base_pct": 0.25,
            "atr_weight": 0.75, "confirmation_bars": 1,
        },
        "trend_weak_up": {
            "atr_multiplier": 1.5, "base_pct": 0.30,
            "atr_weight": 0.55, "confirmation_bars": 2,
        },
        "trend_weak_dn": {
            "atr_multiplier": 1.5, "base_pct": 0.30,
            "atr_weight": 0.55, "confirmation_bars": 2,
        },
        "chop_low_vol": {
            "atr_multiplier": 1.2, "base_pct": 0.20,
            "atr_weight": 0.35, "confirmation_bars": 2,
        },
        "chop_high_vol": {
            "atr_multiplier": 2.0, "base_pct": 0.40,
            "atr_weight": 0.85, "confirmation_bars": 3,
        },
        "volatile": {
            "atr_multiplier": 2.5, "base_pct": 0.50,
            "atr_weight": 0.90, "confirmation_bars": 2,
        },
        "unknown": {
            "atr_multiplier": 1.5, "base_pct": 0.30,
            "atr_weight": 0.50, "confirmation_bars": 2,
        },
    }

    def __init__(self, db: "PivotParameterDB") -> None:
        self._db = db

    def recommend(
        self,
        symbol: str,
        regime: str,
        indicator_type: str = "hybrid_adaptive_pivot",
        lookback_days: int = 30,
        min_sample: int = 3,
    ) -> dict:
        """DB 조회 → 부족 시 폴백 → Config dict 반환."""
        result = self._db.query_best_parameters_session(
            symbol=symbol,
            dominant_regime=regime,
            indicator_type=indicator_type,
            lookback_days=lookback_days,
            min_pivots=5,
            min_sample=min_sample,
        )

        if result and result.get("sample_count", 0) >= min_sample:
            return self._db_result_to_config(result)

        # 폴백: 레짐 하드코딩값
        fallback = self.REGIME_FALLBACK.get(regime, self.REGIME_FALLBACK["unknown"])
        _logger.info(
            "[ParameterRecommender] DB 샘플 부족(%s) → 폴백 사용: regime=%s",
            result.get("sample_count", 0), regime
        )
        return fallback

    @staticmethod
    def _db_result_to_config(row: dict) -> dict:
        """DB 조회 행 → Config dict 변환 (필드명 정규화 포함)."""
        return {
            "atr_multiplier":   row.get("atr_multiplier_bin", 1.5),
            "base_pct":         row.get("base_pct_bin", 0.30),
            "atr_weight":       row.get("atr_weight_bin", 0.50),
            "base_multiplier":  row.get("base_multiplier_bin", 2.0),
            "confirmation_bars": int(row.get("confirmation_bars", 2)),
            "er_period":        int(row.get("er_period", 10)),
            "min_wave_pct":     row.get("min_wave_pct_bin", 0.15),
            # 진단 정보
            "_source":          "db",
            "_sample_count":    row.get("sample_count", 0),
            "_composite_score": row.get("composite_score", 0.0),
        }
