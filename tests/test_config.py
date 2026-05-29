"""config.py 단위 테스트.

NW-TST-04: AppConfig 검증 로직, _get() 헬퍼, 환경변수 우선 적용,
파일 미존재 처리 등 핵심 동작을 검증한다.
"""

from __future__ import annotations

import json
import tempfile
import pytest


# ──────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────

def _write_config(data: dict) -> str:
    """임시 config.json 파일을 작성하고 경로를 반환한다."""
    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, tf)
    tf.flush()
    tf.close()
    return tf.name


def _minimal_config(**overrides) -> dict:
    """유효한 최소 config dict 를 반환한다."""
    base = {
        "prediction": {
            "buy_threshold": 0.62,
            "sell_threshold": 0.38,
            "transformer_weight": 0.5,
        }
    }
    base.update(overrides)
    return base


# ──────────────────────────────────────────────────────────────────
# 테스트 1: buy_threshold < sell_threshold → ValueError
# ──────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_validate_buy_sell_threshold_inverted() -> None:
    """buy_threshold <= sell_threshold 이면 ValueError가 발생하는지 검증.

    from_file()이 내부적으로 validate()를 호출하므로,
    잘못된 임계값을 담은 파일 로드 자체가 ValueError를 발생시킨다.
    """
    from config import AppConfig

    with pytest.raises(ValueError, match="buy_threshold"):
        AppConfig.from_file.__func__(
            AppConfig,
            _write_config({
                "prediction": {
                    "buy_threshold": 0.35,   # < sell_threshold → ValueError
                    "sell_threshold": 0.65,
                }
            })
        )


@pytest.mark.unit
def test_validate_equal_thresholds_raises() -> None:
    """buy_threshold == sell_threshold 일 때도 ValueError가 발생하는지 검증."""
    from config import AppConfig

    cfg = AppConfig._from_dict({
        "prediction": {
            "buy_threshold": 0.50,
            "sell_threshold": 0.50,
        }
    })

    with pytest.raises(ValueError, match="threshold"):
        cfg.validate()


# ──────────────────────────────────────────────────────────────────
# 테스트 2: prediction 섹션 값이 루트보다 우선
# ──────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_pred_cfg_overrides_root_level() -> None:
    """prediction 섹션 값이 루트 레벨보다 우선 적용되는지 검증 (ARC-04 _get() 헬퍼)."""
    from config import AppConfig

    path = _write_config({
        "buy_threshold": 0.70,           # 루트 레벨 (낮은 우선순위)
        "prediction": {
            "buy_threshold": 0.65,       # prediction 섹션 (높은 우선순위)
            "sell_threshold": 0.35,
        }
    })
    cfg = AppConfig.from_file(path)
    assert abs(cfg.prediction.buy_threshold - 0.65) < 1e-9, \
        f"prediction 섹션 우선 미적용: {cfg.prediction.buy_threshold}"


# ──────────────────────────────────────────────────────────────────
# 테스트 3: 파일 미존재 → 기본값으로 fallback
# ──────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_from_file_missing_returns_defaults() -> None:
    """존재하지 않는 config 파일 로드 시 기본값 AppConfig를 반환하는지 검증.

    from_file 은 파일 없을 때 FileNotFoundError 대신 기본값으로 graceful fallback 한다.
    """
    from config import AppConfig

    cfg = AppConfig.from_file("/tmp/nonexistent_config_12345.json")
    assert cfg is not None
    assert isinstance(cfg.prediction.buy_threshold, float)
    assert isinstance(cfg.prediction.sell_threshold, float)
    assert cfg.prediction.buy_threshold > cfg.prediction.sell_threshold


# ──────────────────────────────────────────────────────────────────
# 테스트 4: 환경변수 EBEST_APPKEY 가 config.json보다 우선 적용
# ──────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_env_var_overrides_file_ebest_key(monkeypatch) -> None:
    """EBEST_APPKEY 환경변수가 config.json 값보다 우선 적용되는지 검증."""
    from ebestapi.api import _get_ebest_keys

    path = _write_config({
        "ebest": {
            "appkey": "from_file_key",
            "appsecretkey": "from_file_secret",
        }
    })

    monkeypatch.setenv("EBEST_APPKEY", "env_key_123")
    monkeypatch.setenv("EBEST_APPSECRET", "env_secret_456")

    appkey, appsecret = _get_ebest_keys(config_path=path)

    assert appkey == "env_key_123", f"환경변수 우선 미적용: appkey={appkey!r}"
    assert appsecret == "env_secret_456", f"환경변수 우선 미적용: appsecret={appsecret!r}"


@pytest.mark.unit
def test_file_fallback_when_no_env_var(monkeypatch) -> None:
    """환경변수 없을 때 config.json 값이 사용되는지 검증."""
    from ebestapi.api import _get_ebest_keys

    path = _write_config({
        "ebest": {
            "appkey": "file_key_abc",
            "appsecretkey": "file_secret_xyz",
        }
    })

    monkeypatch.delenv("EBEST_APPKEY", raising=False)
    monkeypatch.delenv("EBEST_APPSECRET", raising=False)
    monkeypatch.delenv("EBEST_APP_KEY", raising=False)
    monkeypatch.delenv("EBEST_APP_SECRET", raising=False)

    appkey, appsecret = _get_ebest_keys(config_path=path)

    assert appkey == "file_key_abc", f"파일 값 미적용: appkey={appkey!r}"
    assert appsecret == "file_secret_xyz", f"파일 값 미적용: appsecret={appsecret!r}"


# ──────────────────────────────────────────────────────────────────
# 테스트 5: 유효한 config 로드 성공 (smoke test)
# ──────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_valid_config_loads_successfully() -> None:
    """정상적인 config 파일이 오류 없이 로드되는지 smoke 검증."""
    from config import AppConfig

    path = _write_config(_minimal_config())
    cfg = AppConfig.from_file(path)
    assert cfg is not None
    # validate() 가 예외 없이 통과해야 함
    cfg.validate()
