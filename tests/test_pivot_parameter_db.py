"""PivotParameterDB 테스트."""

import tempfile
import os
from datetime import datetime, timedelta
from prediction.pivot_parameter_db import PivotParameterDB


class TestPivotParameterDBInit:
    """PivotParameterDB 초기화 테스트."""

    def test_db_initialization_creates_tables(self):
        """DB 초기화 시 테이블 생성."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_params.db")
            db = PivotParameterDB(db_path)
            
            # 테이블 존재 확인
            cursor = db.conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pivot_parameters_daily'"
            )
            result = cursor.fetchone()
            
            assert result is not None
            assert result[0] == "pivot_parameters_daily"
            
            db.close()

    def test_db_initialization_creates_indexes(self):
        """DB 초기화 시 인덱스 생성."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_params.db")
            db = PivotParameterDB(db_path)
            
            # 인덱스 존재 확인
            cursor = db.conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='pivot_parameters_daily'"
            )
            indexes = [row[0] for row in cursor.fetchall()]
            
            assert "idx_pivot_params_date" in indexes
            assert "idx_pivot_params_symbol" in indexes
            assert "idx_pivot_params_indicator" in indexes
            assert "idx_pivot_params_structure" in indexes
            
            db.close()


class TestSaveDailyParameters:
    """일일 파라미터 저장 테스트."""

    def test_save_daily_parameters_basic(self):
        """기본 일일 파라미터 저장."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_params.db")
            db = PivotParameterDB(db_path)
            
            try:
                db.save_daily_parameters(
                    date="2026-05-19",
                    symbol="KP200 선물",
                    indicator_type="adaptive_zigzag",
                    config={
                        "atr_multiplier": 1.5,
                        "atr_period": 14,
                        "base_pct": 0.3,
                        "atr_weight": 0.5,
                    },
                    performance_metrics={
                        "total_pivots": 10,
                        "confirmed_pivots": 8,
                        "cancelled_pivots": 2,
                        "pivot_confirmation_rate": 0.8,
                    },
                    market_state={
                        "market_structure": "uptrend",
                        "avg_atr": 2.5,
                    }
                )
                
                # 저장 확인
                cursor = db.conn.cursor()
                cursor.execute(
                    "SELECT * FROM pivot_parameters_daily WHERE date='2026-05-19' AND symbol='KP200 선물'"
                )
                result = cursor.fetchone()
                
                assert result is not None
                assert result[1] == "2026-05-19"  # date
                assert result[2] == "KP200 선물"  # symbol
                assert result[3] == "adaptive_zigzag"  # indicator_type
                assert result[4] == 1.5  # atr_multiplier
                assert result[5] == 14  # atr_period
                assert result[8] == 0.3  # base_pct (index 8)
                assert result[10] == 0.5  # atr_weight (index 10)
            finally:
                db.close()

    def test_save_daily_parameters_with_null_values(self):
        """None 값이 있는 파라미터 저장."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_params.db")
            db = PivotParameterDB(db_path)
            
            try:
                db.save_daily_parameters(
                    date="2026-05-19",
                    symbol="KOSPI 지수",
                    indicator_type="adaptive_zigzag",
                    config={
                        "atr_multiplier": None,  # None 값
                        "atr_period": 14,
                        "base_pct": 0.3,
                    },
                    performance_metrics={
                        "total_pivots": 5,
                    },
                    market_state={
                        "market_structure": "ranging",
                    }
                )
                
                # 저장 확인
                cursor = db.conn.cursor()
                cursor.execute(
                    "SELECT atr_multiplier FROM pivot_parameters_daily WHERE symbol='KOSPI 지수'"
                )
                result = cursor.fetchone()
                
                assert result is not None
                assert result[0] is None  # None 값 저장 확인
            finally:
                db.close()


class TestQueryBestParameters:
    """최적 파라미터 조회 테스트."""

    def test_query_best_parameters_basic(self):
        """기본 최적 파라미터 조회."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_params.db")
            db = PivotParameterDB(db_path)
            
            try:
                # 테스트 데이터 저장
                for i in range(5):
                    db.save_daily_parameters(
                        date="2026-05-19",
                        symbol="KP200 선물",
                        indicator_type="adaptive_zigzag",
                        config={
                            "atr_multiplier": 1.5,
                            "base_pct": 0.3,
                            "atr_weight": 0.5,
                        },
                        performance_metrics={
                            "total_pivots": 10,
                            "pivot_confirmation_rate": 0.7 + i * 0.05,
                        },
                        market_state={
                            "market_structure": "uptrend",
                        }
                    )
                
                # 최적 파라미터 조회
                best_params = db.query_best_parameters(
                    symbol="KP200 선물",
                    market_structure="uptrend",
                    lookback_days=30,
                )
                
                assert best_params is not None
                assert "atr_multiplier" in best_params
                assert "base_pct" in best_params
                assert "atr_weight" in best_params
                assert "avg_confirmation_rate" in best_params
                assert "sample_count" in best_params
            finally:
                db.close()

    def test_query_best_parameters_no_data(self):
        """데이터 없을 때 빈 결과 반환."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_params.db")
            db = PivotParameterDB(db_path)
            
            try:
                best_params = db.query_best_parameters(
                    symbol="KP200 선물",
                    market_structure="uptrend",
                    lookback_days=30,
                )
                
                assert best_params == {}
            finally:
                db.close()

    def test_query_best_parameters_min_pivots_filter(self):
        """최소 피봇 수 필터."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_params.db")
            db = PivotParameterDB(db_path)
            
            try:
                # 적은 피봇 수 데이터
                db.save_daily_parameters(
                    date="2026-05-19",
                    symbol="KP200 선물",
                    indicator_type="adaptive_zigzag",
                    config={"atr_multiplier": 1.5},
                    performance_metrics={"total_pivots": 2},  # min_pivots 미만
                    market_state={"market_structure": "uptrend"},
                )
                
                # 많은 피봇 수 데이터
                db.save_daily_parameters(
                    date="2026-05-20",
                    symbol="KP200 선물",
                    indicator_type="adaptive_zigzag",
                    config={"atr_multiplier": 2.0},
                    performance_metrics={"total_pivots": 10},  # min_pivots 이상
                    market_state={"market_structure": "uptrend"},
                )
                
                best_params = db.query_best_parameters(
                    symbol="KP200 선물",
                    market_structure="uptrend",
                    lookback_days=30,
                    min_pivots=5,
                )
                
                assert best_params is not None
                assert best_params["atr_multiplier"] == 2.0  # 많은 피봇 수 데이터 선택
            finally:
                db.close()


class TestAnalyzeParameterPerformance:
    """파라미터 성능 분석 테스트."""

    def test_analyze_parameter_performance_basic(self):
        """기본 파라미터 성능 분석."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_params.db")
            db = PivotParameterDB(db_path)
            
            try:
                # 테스트 데이터 저장
                for i in range(3):
                    db.save_daily_parameters(
                        date="2026-05-19",
                        symbol="KP200 선물",
                        indicator_type="adaptive_zigzag",
                        config={"atr_multiplier": 1.5},
                        performance_metrics={
                            "total_pivots": 10,
                            "confirmed_pivots": 8,
                            "pivot_confirmation_rate": 0.8,
                        },
                        market_state={"market_structure": "uptrend"},
                    )
                
                analysis = db.analyze_parameter_performance(
                    symbol="KP200 선물",
                    start_date="2026-05-01",
                    end_date="2026-05-31",
                )
                
                assert analysis is not None
                assert "symbol" in analysis
                assert "overall_avg_confirmation_rate" in analysis
                assert "by_market_structure" in analysis
                assert len(analysis["by_market_structure"]) > 0
            finally:
                db.close()

    def test_analyze_parameter_performance_no_data(self):
        """데이터 없을 때 빈 결과."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_params.db")
            db = PivotParameterDB(db_path)
            
            try:
                analysis = db.analyze_parameter_performance(
                    symbol="KP200 선물",
                    start_date="2026-05-01",
                    end_date="2026-05-31",
                )
                
                assert analysis is not None
                assert analysis["overall_avg_confirmation_rate"] == 0.0
                assert len(analysis["by_market_structure"]) == 0
            finally:
                db.close()


class TestQueryParameterDistribution:
    """파라미터 분포 조회 테스트."""

    def test_query_parameter_distribution_basic(self):
        """기본 파라미터 분포 조회."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_params.db")
            db = PivotParameterDB(db_path)
            
            try:
                # 다양한 atr_multiplier 값 저장
                for mult in [1.0, 1.5, 2.0]:
                    db.save_daily_parameters(
                        date="2026-05-19",
                        symbol="KP200 선물",
                        indicator_type="adaptive_zigzag",
                        config={"atr_multiplier": mult},
                        performance_metrics={"total_pivots": 10},
                        market_state={"market_structure": "uptrend"},
                    )
                
                distribution = db.query_parameter_distribution(
                    symbol="KP200 선물",
                    parameter_name="atr_multiplier",
                )
                
                assert distribution is not None
                assert "parameter_name" in distribution
                assert "weighted_avg" in distribution
                assert "min" in distribution
                assert "max" in distribution
                assert "distribution" in distribution
                assert len(distribution["distribution"]) == 3
            finally:
                db.close()

    def test_query_parameter_distribution_invalid_parameter(self):
        """유효하지 않은 파라미터명."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_params.db")
            db = PivotParameterDB(db_path)
            
            try:
                distribution = db.query_parameter_distribution(
                    symbol="KP200 선물",
                    parameter_name="invalid_param",
                )
                
                assert distribution == {}
            finally:
                db.close()


class TestGetAllDates:
    """날짜 목록 조회 테스트."""

    def test_get_all_dates_basic(self):
        """기본 날짜 목록 조회."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_params.db")
            db = PivotParameterDB(db_path)
            
            try:
                # 다양한 날짜 데이터 저장
                for date in ["2026-05-18", "2026-05-19", "2026-05-20"]:
                    db.save_daily_parameters(
                        date=date,
                        symbol="KP200 선물",
                        indicator_type="adaptive_zigzag",
                        config={"atr_multiplier": 1.5},
                        performance_metrics={"total_pivots": 10},
                        market_state={"market_structure": "uptrend"},
                    )
                
                dates = db.get_all_dates(symbol="KP200 선물")
                
                assert len(dates) == 3
                assert "2026-05-18" in dates
                assert "2026-05-19" in dates
                assert "2026-05-20" in dates
            finally:
                db.close()

    def test_get_all_dates_symbol_filter(self):
        """심볼 필터."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_params.db")
            db = PivotParameterDB(db_path)
            
            try:
                # KP200 데이터
                db.save_daily_parameters(
                    date="2026-05-19",
                    symbol="KP200 선물",
                    indicator_type="adaptive_zigzag",
                    config={"atr_multiplier": 1.5},
                    performance_metrics={"total_pivots": 10},
                    market_state={"market_structure": "uptrend"},
                )
                
                # KOSPI 데이터
                db.save_daily_parameters(
                    date="2026-05-19",
                    symbol="KOSPI 지수",
                    indicator_type="adaptive_zigzag",
                    config={"atr_multiplier": 1.0},
                    performance_metrics={"total_pivots": 5},
                    market_state={"market_structure": "ranging"},
                )
                
                kp200_dates = db.get_all_dates(symbol="KP200 선물")
                kospi_dates = db.get_all_dates(symbol="KOSPI 지수")
                
                assert len(kp200_dates) == 1
                assert len(kospi_dates) == 1
            finally:
                db.close()


class TestDeleteOldRecords:
    """오래된 레코드 삭제 테스트."""

    def test_delete_old_records_basic(self):
        """기본 오래된 레코드 삭제."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_params.db")
            db = PivotParameterDB(db_path)
            
            try:
                # 오래된 데이터
                old_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
                db.save_daily_parameters(
                    date=old_date,
                    symbol="KP200 선물",
                    indicator_type="adaptive_zigzag",
                    config={"atr_multiplier": 1.5},
                    performance_metrics={"total_pivots": 10},
                    market_state={"market_structure": "uptrend"},
                )
                
                # 최신 데이터
                new_date = datetime.now().strftime("%Y-%m-%d")
                db.save_daily_parameters(
                    date=new_date,
                    symbol="KP200 선물",
                    indicator_type="adaptive_zigzag",
                    config={"atr_multiplier": 1.5},
                    performance_metrics={"total_pivots": 10},
                    market_state={"market_structure": "uptrend"},
                )
                
                # 365일 이전 데이터 삭제
                db.delete_old_records(days_to_keep=365)
                
                # 오래된 데이터 삭제 확인
                cursor = db.conn.cursor()
                cursor.execute(
                    f"SELECT COUNT(*) FROM pivot_parameters_daily WHERE date='{old_date}'"
                )
                old_count = cursor.fetchone()[0]
                
                cursor.execute(
                    f"SELECT COUNT(*) FROM pivot_parameters_daily WHERE date='{new_date}'"
                )
                new_count = cursor.fetchone()[0]
                
                assert old_count == 0  # 오래된 데이터 삭제됨
                assert new_count == 1  # 최신 데이터 유지됨
            finally:
                db.close()
