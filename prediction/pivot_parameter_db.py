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
        as_of_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """세션·레짐 단위 최적 파라미터 조회 (복합 점수 기반).
        
        Args:
            symbol: 심볼명
            dominant_regime: 레짐 (trend_strong/chop/volatile/mixed), None이면 전체
            indicator_type: 지표 타입, None이면 전체
            lookback_days: 조회 기간 (일)
            min_pivots: 최소 피봇 수
            min_sample: 최소 샘플 수
            as_of_date: 기준일(YYYY-MM-DD). 지정 시 '이 날짜 미만(<)'의 데이터만
                조회하여 워크포워드 룩어헤드 누수를 차단한다. None이면 현재시각 기준.
        
        Returns:
            최적 파라미터 딕셔너리
        """
        cursor = self.conn.cursor()
        
        # 날짜 범위 계산 (as_of_date 지정 시 그 날짜 '미만'만 → 룩어헤드 차단)
        if as_of_date:
            try:
                end_date = datetime.strptime(as_of_date, "%Y-%m-%d")
            except ValueError:
                _logger.warning("[PIVOT_PARAM_DB] as_of_date 파싱 실패(%s), 현재시각 사용", as_of_date)
                end_date = datetime.now()
        else:
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
              AND date < ?
              AND total_pivots >= ?
        """
        params = [symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), min_pivots]
        
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
        
        # pivot_parameters_daily + pivot_parameters_session 양쪽에서 날짜 수집
        if symbol:
            cursor.execute(
                """
                SELECT DISTINCT date FROM (
                    SELECT date FROM pivot_parameters_daily WHERE symbol = ?
                    UNION
                    SELECT date FROM pivot_parameters_session WHERE symbol = ?
                ) ORDER BY date DESC
                """,
                (symbol, symbol),
            )
        else:
            cursor.execute(
                """
                SELECT DISTINCT date FROM (
                    SELECT date FROM pivot_parameters_daily
                    UNION
                    SELECT date FROM pivot_parameters_session
                ) ORDER BY date DESC
                """
            )
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
        as_of_date: Optional[str] = None,
    ) -> dict:
        """DB 조회 → 부족 시 폴백 → Config dict 반환.

        as_of_date 지정 시 그 날짜 '미만'의 데이터만으로 추천한다(룩어헤드 차단).
        """
        result = self._db.query_best_parameters_session(
            symbol=symbol,
            dominant_regime=regime,
            indicator_type=indicator_type,
            lookback_days=lookback_days,
            min_pivots=5,
            min_sample=min_sample,
            as_of_date=as_of_date,
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


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward 평가기 (신규 추가)
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardEvaluator:
    """DB 추천 파라미터의 실질 효과를 Walk-forward 방식으로 검증한다.

    평가 구조
    ---------
    훈련 윈도우 (test_date '미만' lookback_days) → ParameterRecommender 추천
    테스트 일 (test_date)        → composite_score 비교
    슬라이딩                    → 하루씩 전진하며 N회 반복

    [누수 수정] 추천은 항상 as_of_date=test_date 로 호출되어 test_date 이전
    데이터만 사용한다(룩어헤드 차단).

    두 가지 평가 모드
    -----------------
    - real 모드 (evaluator_fn 주입): test_date 봉 데이터에 각 파라미터로 검출을
      '재실행'해 실제 성능을 측정한다. → 순환 강화/효과를 실증.
    - estimate 모드 (evaluator_fn=None): 검출 재실행 없이 동일 지표에 선형
      surrogate 보정만 적용한다. 이는 파라미터 차이의 닫힌 함수이므로 실증이
      아니며, verdict 에 'ESTIMATE_ONLY' 로 명시한다.

    비교 대상
    ---------
    A) DB 추천 (ParameterRecommender.recommend, as_of_date=test_date)
    B) REGIME_FALLBACK 하드코딩 폴백
    C) 고정 파라미터 (base_pct=0.3, atr_weight=0.5)

    판정 기준 (real 모드)
    ---------
    improvement_vs_fallback > 0.03  → DB 추천 유효
    improvement_vs_fallback < 0.00  → 순환 강화 의심
    coverage_rate           ≥ 0.80  → 샘플 충분
    stability_cv            < 0.20  → 파라미터 안정
    """

    # 고정 파라미터 베이스라인 C
    FIXED_BASELINE: dict = {
        "atr_multiplier":   1.5,
        "base_pct":         0.30,
        "atr_weight":       0.50,
        "confirmation_bars": 2,
        "er_period":        10,
        "min_wave_pct":     0.15,
    }

    def __init__(
        self,
        db: "PivotParameterDB",
        symbol: str = "KP200 선물",
        evaluator_fn=None,
    ) -> None:
        """초기화.

        Parameters
        ----------
        evaluator_fn: callable | None
            실측 평가 함수. 시그니처:
                evaluator_fn(date: str, params: dict) -> dict | None
            지정 시, 해당 날짜의 실제 봉 데이터에 params 로 피봇 검출을
            '재실행'하여 성능 지표를 돌려줘야 한다. 반환 dict 키:
                pivot_confirmation_rate, avg_lag_bars, pivot_quality_score,
                alternation_rate, false_pivot_rate
            None 이면 검출 재실행 없이 선형 surrogate(_estimate_composite)로
            '추정'만 수행하며, 결과는 실증이 아님을 명시한다.
        """
        self._db = db
        self._symbol = symbol
        self._recommender = ParameterRecommender(db)
        self._evaluator_fn = evaluator_fn

    # ── 메인 실행 ────────────────────────────────────────────────────────────

    def run(
        self,
        lookback_days: int = 30,
        test_days: int = 20,
        indicator_type: str = "hybrid_adaptive_pivot",
    ) -> dict:
        """Walk-forward 검증을 실행하고 결과 딕셔너리를 반환한다.

        Parameters
        ----------
        lookback_days:
            훈련 윈도우 크기 (일). 이 기간의 DB 데이터로 파라미터 추천.
        test_days:
            슬라이딩 반복 횟수 (일). DB에 저장된 최근 날짜부터 역방향으로.
        indicator_type:
            평가 대상 지표 타입.

        Returns
        -------
        dict:
            improvement_vs_fallback, improvement_vs_fixed,
            coverage_rate, stability_cv, daily_results 포함.
        """
        all_dates = self._db.get_all_dates(self._symbol)
        if len(all_dates) < lookback_days + 1:
            _logger.warning(
                "[WalkForward] 데이터 부족: %d일 필요, %d일 보유",
                lookback_days + 1, len(all_dates),
            )
            return self._empty_result("데이터 부족")

        # 날짜 오름차순 정렬
        sorted_dates = sorted(all_dates)
        # 테스트 구간: 마지막 test_days개
        test_date_list = sorted_dates[lookback_days:][-test_days:]
        if not test_date_list:
            return self._empty_result("테스트 날짜 없음")

        daily_results = []
        param_history: dict[str, list] = {
            "atr_multiplier": [], "base_pct": [], "atr_weight": [],
        }
        fallback_used = 0

        for test_date in test_date_list:
            result = self._run_single_day_comparison(
                test_date, lookback_days, indicator_type,
            )
            if result is None:
                continue
            daily_results.append(result)

            # 파라미터 이력 누적 (안정성 계산용)
            for key in param_history:
                v = result.get(f"db_param_{key}")
                if v is not None:
                    param_history[key].append(v)

            if result.get("db_used_fallback", False):
                fallback_used += 1

        if not daily_results:
            return self._empty_result("비교 가능한 날 없음")

        # 집계
        improvements_f = [r["improvement_vs_fallback"] for r in daily_results]
        improvements_x = [r["improvement_vs_fixed"]    for r in daily_results]
        n = len(daily_results)

        avg_improvement_f  = float(sum(improvements_f) / n)
        avg_improvement_x  = float(sum(improvements_x) / n)
        win_rate_vs_fallback = float(sum(1 for v in improvements_f if v > 0) / n)
        coverage_rate        = float(1.0 - fallback_used / n)

        # 파라미터 안정성 CV (변동 계수)
        stability_cv = self._calc_stability_cv(param_history)

        mode = "real" if self._evaluator_fn is not None else "estimate"
        is_estimate = (mode == "estimate")

        summary = {
            "symbol":                  self._symbol,
            "mode":                    mode,
            "lookback_days":           lookback_days,
            "test_days":               n,
            "avg_improvement_vs_fallback": round(avg_improvement_f, 4),
            "avg_improvement_vs_fixed":    round(avg_improvement_x, 4),
            "win_rate_vs_fallback":    round(win_rate_vs_fallback, 4),
            "coverage_rate":           round(coverage_rate, 4),
            "stability_cv":            round(stability_cv, 4),
            "verdict":                 self._verdict(
                avg_improvement_f, coverage_rate, stability_cv, is_estimate,
            ),
            "daily_results":           daily_results,
        }

        _logger.info(
            "[WalkForward] 완료(mode=%s): 개선률=%.3f coverage=%.2f CV=%.3f 판정=%s",
            mode, avg_improvement_f, coverage_rate, stability_cv, summary["verdict"],
        )
        return summary

    # ── 단일 날짜 비교 ────────────────────────────────────────────────────────

    def _run_single_day_comparison(
        self,
        test_date: str,
        lookback_days: int,
        indicator_type: str,
    ) -> dict | None:
        """test_date 에 대해 3가지 파라미터 소스의 composite_score 를 비교.

        - 파라미터 추천은 as_of_date=test_date 로 호출 → test_date 이전만 사용.
        - evaluator_fn 이 있으면 test_date 봉에 검출을 재실행한 '실측' 점수,
          없으면 선형 surrogate '추정' 점수를 사용한다(실증 아님 표기).
        """
        cursor = self._db.conn.cursor()

        # test_date 레코드 조회 (레짐/실제 성능 지표 확보)
        cursor.execute(
            """
            SELECT dominant_regime, composite_score,
                   pivot_confirmation_rate, avg_lag_bars,
                   pivot_quality_score, alternation_rate, false_pivot_rate,
                   atr_multiplier_bin, base_pct_bin, atr_weight_bin
            FROM pivot_parameters_session
            WHERE symbol = ? AND date = ?
            ORDER BY composite_score DESC
            LIMIT 1
            """,
            (self._symbol, test_date),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        (
            regime, actual_composite,
            conf_rate, lag_bars, quality, alt_rate, fp_rate,
            db_atr_mult, db_base_pct, db_atr_weight,
        ) = row

        if actual_composite is None:
            return None

        regime = regime or "unknown"

        # [누수 수정] 추천은 test_date '미만' 데이터만 사용
        db_params = self._recommender.recommend(
            symbol=self._symbol,
            regime=regime,
            indicator_type=indicator_type,
            lookback_days=lookback_days,
            as_of_date=test_date,
        )
        db_used_fallback = db_params.get("_source") != "db"

        fallback_params = ParameterRecommender.REGIME_FALLBACK.get(
            regime, ParameterRecommender.REGIME_FALLBACK["unknown"],
        )

        if self._evaluator_fn is not None:
            # ── real 모드: test_date 봉에 검출 재실행 ──────────────────────
            is_estimate = False
            score_db       = self._real_composite(test_date, db_params)
            score_fallback = self._real_composite(test_date, fallback_params)
            score_fixed    = self._real_composite(test_date, self.FIXED_BASELINE)
            # 셋 중 하나라도 측정 실패면 비교 불가 → 해당 날 제외
            if score_db is None or score_fallback is None or score_fixed is None:
                return None
        else:
            # ── estimate 모드: surrogate (실증 아님) ──────────────────────
            is_estimate = True
            score_db       = self._estimate_composite(db_params,       conf_rate, lag_bars, quality, alt_rate, fp_rate)
            score_fallback = self._estimate_composite(fallback_params,  conf_rate, lag_bars, quality, alt_rate, fp_rate)
            score_fixed    = self._estimate_composite(self.FIXED_BASELINE, conf_rate, lag_bars, quality, alt_rate, fp_rate)

        return {
            "date":                     test_date,
            "regime":                   regime,
            "is_estimate":              is_estimate,
            "actual_composite_score":   round(float(actual_composite), 4),
            "score_db":                 round(score_db, 4),
            "score_fallback":           round(score_fallback, 4),
            "score_fixed":              round(score_fixed, 4),
            "improvement_vs_fallback":  round(score_db - score_fallback, 4),
            "improvement_vs_fixed":     round(score_db - score_fixed, 4),
            "db_used_fallback":         db_used_fallback,
            "db_param_atr_multiplier":  db_params.get("atr_multiplier"),
            "db_param_base_pct":        db_params.get("base_pct"),
            "db_param_atr_weight":      db_params.get("atr_weight"),
        }

    # ── 실측 composite_score (real 모드) ─────────────────────────────────────

    def _real_composite(self, test_date: str, params: dict):
        """evaluator_fn 으로 test_date 에 params 검출을 재실행해 composite 산출.

        Returns
        -------
        float | None : 측정 실패(데이터 없음 등) 시 None.
        """
        try:
            metrics = self._evaluator_fn(test_date, params)
        except Exception as e:
            _logger.warning("[WalkForward] evaluator_fn 실패(%s, %s): %s", test_date, params, e)
            return None
        if not metrics:
            return None
        return self._db._calc_composite_score(
            confirmation_rate = float(metrics.get("pivot_confirmation_rate", 0.0)),
            avg_lag_bars      = float(metrics.get("avg_lag_bars", 10.0)),
            pivot_quality     = float(metrics.get("pivot_quality_score", 0.0)),
            alternation_rate  = float(metrics.get("alternation_rate", 0.0)),
            false_pivot_rate  = float(metrics.get("false_pivot_rate", 0.0)),
        )

    # ── composite_score 추정 ─────────────────────────────────────────────────

    @staticmethod
    def _estimate_composite(
        params: dict,
        conf_rate: float,
        lag_bars:  float,
        quality:   float,
        alt_rate:  float,
        fp_rate:   float,
    ) -> float:
        """[surrogate] 파라미터 소스별 composite_score '추정' (실증 아님).

        주의: 검출을 재실행하지 않고, 동일 test_date 지표에 confirmation_bars /
        atr_weight 의 선형 보정만 적용한다. 따라서 score 차이는 파라미터 차이의
        닫힌 함수이며 시장 반응을 반영하지 못한다. 실증이 필요하면
        WalkForwardEvaluator(evaluator_fn=...) 의 real 모드를 사용할 것.

        - confirmation_bars 증가 → 래그 증가, 오탐 감소
        - atr_weight 높음 → 변동성 장에서 확정률 증가 (가정)
        """
        cb = float(params.get("confirmation_bars", 2))
        aw = float(params.get("atr_weight", 0.5))

        # confirmation_bars 보정: 기준(2봉) 대비 래그 증감
        lag_adj   = lag_bars  + (cb - 2.0) * 1.5   # 봉당 1.5 래그 추가
        conf_adj  = max(0.0, min(1.0, conf_rate - (cb - 2.0) * 0.03))  # 봉당 3% 감소

        # atr_weight 보정: 0.5 기준으로 ±10% 확정률 영향
        conf_adj  = max(0.0, min(1.0, conf_adj + (aw - 0.5) * 0.10))

        lag_score  = max(0.0, 1.0 - lag_adj / 20.0)
        fp_adj     = max(0.0, min(1.0, fp_rate - (cb - 2.0) * 0.02))
        fp_score   = 1.0 - fp_adj

        return (
            conf_adj  * 0.35
            + lag_score * 0.25
            + quality   * 0.20
            + alt_rate  * 0.10
            + fp_score  * 0.10
        )

    # ── 파라미터 안정성 ──────────────────────────────────────────────────────

    @staticmethod
    def _calc_stability_cv(param_history: dict) -> float:
        """연속 날의 추천 파라미터 변동 계수(CV) 평균을 반환한다.

        CV = std / mean. 낮을수록 추천이 안정적.
        목표: CV < 0.20
        """
        cvs = []
        for key, values in param_history.items():
            if len(values) < 2:
                continue
            import statistics
            try:
                mean = statistics.mean(values)
                std  = statistics.stdev(values)
                if mean > 1e-10:
                    cvs.append(std / mean)
            except Exception:
                pass
        return float(sum(cvs) / len(cvs)) if cvs else 0.0

    # ── 커버리지 측정 ────────────────────────────────────────────────────────

    def measure_coverage(
        self,
        regimes: list | None = None,
        lookback_days: int = 30,
        min_sample: int = 3,
    ) -> dict:
        """레짐별 DB 샘플 커버리지를 측정한다.

        Parameters
        ----------
        regimes:
            측정할 레짐 목록. None이면 8개 전체.
        lookback_days:
            조회 기간.
        min_sample:
            최소 샘플 수 (미달 시 폴백 사용).

        Returns
        -------
        dict:
            레짐별 sample_count, covered(bool), overall_coverage_rate 포함.
        """
        if regimes is None:
            regimes = list(ParameterRecommender.REGIME_FALLBACK.keys())

        coverage_detail = {}
        covered_count = 0

        for regime in regimes:
            result = self._db.query_best_parameters_session(
                symbol=self._symbol,
                dominant_regime=regime,
                lookback_days=lookback_days,
                min_pivots=5,
                min_sample=min_sample,
            )
            sample_count = result.get("sample_count", 0) if result else 0
            covered = sample_count >= min_sample
            if covered:
                covered_count += 1
            coverage_detail[regime] = {
                "sample_count": sample_count,
                "covered":      covered,
            }

        return {
            "regimes":              coverage_detail,
            "overall_coverage_rate": round(covered_count / len(regimes), 4),
            "covered_regimes":      covered_count,
            "total_regimes":        len(regimes),
        }

    # ── 판정 로직 ────────────────────────────────────────────────────────────

    @staticmethod
    def _verdict(
        improvement: float,
        coverage:    float,
        stability:   float,
        is_estimate: bool = False,
    ) -> str:
        """판정 문자열 반환.

        estimate 모드(검출 재실행 없음)에서는 surrogate 계산이 파라미터 차이의
        닫힌 함수이므로 '유효/순환강화'를 단정하지 않고 ESTIMATE_ONLY 로 표기한다.
        """
        if is_estimate:
            return (
                "ESTIMATE_ONLY — surrogate 추정값(검출 재실행 아님). "
                "실증하려면 evaluator_fn 주입 필요. "
                f"(추정 개선률={improvement:+.3f}, coverage={coverage:.2f})"
            )
        if improvement > 0.03 and coverage >= 0.80 and stability < 0.20:
            return "PASS — DB 추천 유효"
        if improvement > 0.00 and coverage >= 0.60:
            return "MARGINAL — 데이터 추가 누적 필요"
        if improvement < 0.00:
            return "FAIL — 순환 강화 의심, 폴백 우선 권장"
        return "INSUFFICIENT — 샘플 부족"

    def _empty_result(self, reason: str) -> dict:
        return {
            "verdict": f"ERROR — {reason}",
            "mode": "real" if self._evaluator_fn is not None else "estimate",
            "avg_improvement_vs_fallback": 0.0,
            "coverage_rate": 0.0,
            "stability_cv":  0.0,
            "daily_results": [],
        }
