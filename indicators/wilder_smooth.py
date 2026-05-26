"""Wilder smoothing helper.

Internal utility for indicators that use Wilder's RMA (alpha = 1/period).

Warmup stage
------------
  첫 `period` 봉은 버퍼에 누적한 뒤 SMA로 초기화.
  이후 봉은 Wilder EMA (alpha = 1/period) 적용.

[BUG FIX] ready 조건: count >= period (이전 SkyEbest 구현의 count > period 1봉 지연 수정)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WilderRMA:
    """Wilder's Smoothed Moving Average (RMA).

    >>> rma = WilderRMA(period=3)
    >>> for v in [1, 2, 3, 4]:
    ...     _ = rma.update(v)
    >>> rma.ready
    True
    """

    period: int
    value: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        self._buf: list[float] = []

    def reset(self) -> None:
        self.value = 0.0
        self.count = 0
        self._buf = []

    @property
    def ready(self) -> bool:
        # [FIX] count >= period  (SkyEbest는 count > period 로 1봉 지연이 있었음)
        return int(self.count) >= int(self.period)

    def update(self, x: float) -> float:
        self.count += 1
        if int(self.period) <= 1:
            self.value = float(x)
            return float(self.value)

        if not self.ready:
            self._buf.append(float(x))
            if len(self._buf) >= int(self.period):
                self.value = sum(self._buf[-int(self.period):]) / float(self.period)
            else:
                self.value = sum(self._buf) / float(len(self._buf) or 1)
            return float(self.value)

        alpha = 1.0 / float(self.period)
        self.value = float(self.value) * (1.0 - alpha) + float(x) * alpha
        return float(self.value)
