import gzip
import json

from prediction import data_builder


def test_data_builder_load_jsonl_gz_and_restore_compact_prices(tmp_path) -> None:
    p = tmp_path / "ticks_replay_test.jsonl.gz"

    rec = {
        "ts_ms": 0,
        "trcode": "FH0",
        "symbol": "A016XXXX",
        "tick": {
            # compact schema: int(x100)
            "offerho1": 43055,
            "bidho1": 43045,
            "offerrem1": 10,
            "bidrem1": 20,
            "totofferrem": 100,
            "totbidrem": 200,
            "hotime": "130431",
        },
    }

    with gzip.open(str(p), "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    loaded = list(data_builder._load_jsonl(str(p)))
    assert len(loaded) == 1
    tick = loaded[0].get("tick")

    restored = data_builder._restore_compact_prices(tick)
    assert abs(float(restored.get("offerho1")) - 430.55) < 1e-9
    assert abs(float(restored.get("bidho1")) - 430.45) < 1e-9
