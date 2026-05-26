"""백테스팅용 데이터 자동 저장 모듈.

장마감 시 KOSPI 지수와 KP200 선물 분봉 데이터를 자동으로 저장합니다.
"""

from __future__ import annotations

import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd


class BacktestDataSaver:
    """백테스팅용 데이터 자동 저장 클래스."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        market_close_time: str = "15:35",
    ) -> None:
        """초기화.

        Args:
            base_dir: 데이터 저장 기본 디렉토리 (기본: data/backtesting)
            market_close_time: 장마감 시간 (기본: 15:35)
        """
        if base_dir is None:
            base_dir = Path("data/backtesting")
        self.base_dir = Path(base_dir)
        self.market_close_time = market_close_time

        # 디렉토리 구조 생성
        self.kospi_dir = self.base_dir / "kospi"
        self.futures_dir = self.base_dir / "futures"

        self.kospi_dir.mkdir(parents=True, exist_ok=True)
        self.futures_dir.mkdir(parents=True, exist_ok=True)

    def is_market_closed(self, current_time: Optional[datetime] = None) -> bool:
        """장마감 시간이 지났는지 확인.

        Args:
            current_time: 현재 시간 (기본: 현재 시간)

        Returns:
            장마감 여부
        """
        if current_time is None:
            current_time = datetime.now()

        close_time = datetime.strptime(self.market_close_time, "%H:%M").time()
        current_time_only = current_time.time()

        return current_time_only >= close_time

    def get_data_filename(
        self,
        data_source: str,
        timeframe: str,
        data_date: Optional[date] = None,
    ) -> Path:
        """데이터 파일 경로 생성.

        Args:
            data_source: 데이터 소스 (kospi 또는 futures)
            timeframe: 타임프레임 (1m, 5m 등)
            data_date: 데이터 날짜 (기본: 오늘)

        Returns:
            데이터 파일 경로
        """
        if data_date is None:
            data_date = date.today()

        year_dir = self.base_dir / data_source / str(data_date.year)
        year_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{data_date.strftime('%Y-%m-%d')}_{data_source}_{timeframe}.csv"
        return year_dir / filename

    def is_data_already_saved(
        self,
        data_source: str,
        timeframe: str,
        data_date: Optional[date] = None,
    ) -> bool:
        """데이터가 이미 저장되었는지 확인.

        Args:
            data_source: 데이터 소스 (kospi 또는 futures)
            timeframe: 타임프레임 (1m, 5m 등)
            data_date: 데이터 날짜 (기본: 오늘)

        Returns:
            저장 여부
        """
        filepath = self.get_data_filename(data_source, timeframe, data_date)
        return filepath.exists()

    def save_data(
        self,
        df: pd.DataFrame,
        data_source: str,
        timeframe: str,
        data_date: Optional[date] = None,
    ) -> bool:
        """데이터 저장.

        Args:
            df: 저장할 데이터프레임
            data_source: 데이터 소스 (kospi 또는 futures)
            timeframe: 타임프레임 (1m, 5m 등)
            data_date: 데이터 날짜 (기본: 오늘)

        Returns:
            저장 성공 여부
        """
        try:
            filepath = self.get_data_filename(data_source, timeframe, data_date)
            df.to_csv(filepath, index=False)
            print(f"[BacktestDataSaver] 데이터 저장 완료: {filepath}")
            return True
        except Exception as e:
            print(f"[BacktestDataSaver] 데이터 저장 실패: {e}")
            return False

    def save_if_needed(
        self,
        df: pd.DataFrame,
        data_source: str,
        timeframe: str,
        data_date: Optional[date] = None,
        force: bool = False,
    ) -> bool:
        """필요한 경우에만 데이터 저장.

        Args:
            df: 저장할 데이터프레임
            data_source: 데이터 소스 (kospi 또는 futures)
            timeframe: 타임프레임 (1m, 5m 등)
            data_date: 데이터 날짜 (기본: 오늘)
            force: 강제 저장 여부

        Returns:
            저장 여부
        """
        # 장마감 확인
        if not force and not self.is_market_closed():
            return False

        # 이미 저장된 데이터 확인
        if not force and self.is_data_already_saved(data_source, timeframe, data_date):
            return False

        return self.save_data(df, data_source, timeframe, data_date)


# 전역 인스턴스
_saver_instance: Optional[BacktestDataSaver] = None


def get_backtest_data_saver() -> BacktestDataSaver:
    """백테스팅 데이터 저장 인스턴스 가져오기.

    Returns:
        BacktestDataSaver 인스턴스
    """
    global _saver_instance
    if _saver_instance is None:
        _saver_instance = BacktestDataSaver()
    return _saver_instance
