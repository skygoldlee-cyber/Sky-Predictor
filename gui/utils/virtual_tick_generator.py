"""
가상 틱 데이터 생성기

테스트 및 시뮬레이션을 위한 가상 틱 데이터를 생성합니다.
"""

import random
import logging

logger = logging.getLogger(__name__)


class VirtualTickGenerator:
    """가상 틱 데이터 생성기."""

    def __init__(self, base_price: float, volatility: float = 0.001, tick_size: float = 0.05):
        """가상 틱 생성기 초기화.

        Args:
            base_price: 기준 가격
            volatility: 변동성 (가격 변동 비율)
            tick_size: 틱 단위
        """
        self.base_price = base_price
        self.current_price = base_price
        self.volatility = volatility
        self.tick_size = tick_size
        self.trend = 0.0  # 추세 (-1.0 ~ 1.0)
        self.trend_change_prob = 0.05  # 추세 변경 확률
        self._random = random.Random()

    def generate_tick(self) -> tuple:
        """단일 틱 생성.

        Returns:
            (price, volume) 튜플
        """
        # 추세 변경 (확률적)
        if self._random.random() < self.trend_change_prob:
            self.trend = self._random.uniform(-1.0, 1.0) * 0.3  # 추세 강도 제한

        # 가격 변동 (가우시안 + 추세)
        change_pct = self._random.gauss(0, self.volatility) + self.trend * self.volatility
        new_price = self.current_price * (1 + change_pct)

        # 틱 단위 반올림
        new_price = round(new_price / self.tick_size) * self.tick_size

        # 가격 하한/상한 (너무 극단적인 변동 방지)
        max_change = self.current_price * 0.01  # 1% 이상 변동 제한
        new_price = max(self.current_price - max_change, min(self.current_price + max_change, new_price))

        self.current_price = new_price

        # 거래량 (가격 변동이 클수록 거래량 증가)
        volume = int(abs(change_pct) * 10000 * self._random.uniform(0.5, 1.5))
        volume = max(1, volume)

        return new_price, volume

    def generate_ohlc(self, num_ticks: int = 10) -> dict:
        """OHLC 데이터 생성.

        Args:
            num_ticks: 생성할 틱 수

        Returns:
            OHLC 딕셔너리
        """
        prices = [self.current_price]
        volumes = []

        for _ in range(num_ticks):
            price, vol = self.generate_tick()
            prices.append(price)
            volumes.append(vol)

        return {
            'Open': prices[0],
            'High': max(prices),
            'Low': min(prices),
            'Close': prices[-1],
            'Volume': sum(volumes)
        }
