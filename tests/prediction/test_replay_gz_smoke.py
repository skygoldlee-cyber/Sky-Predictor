import gzip
import json

from prediction.pipeline import PredictionPipeline
from app.run_modes import run_replay_mode_with_predictor


def test_replay_jsonl_gz_smoke(tmp_path) -> None:
    # Build a minimal replay log containing FC0 + FH0 so the pipeline can ingest
    # without requiring external services.
    p = tmp_path / "ticks_replay_test.jsonl.gz"

    records = [
        {
            "ts_ms": 0,
            "trcode": "FC0",
            "symbol": "A016XXXX",
            "tick": {
                "price": "430.50",
                "volume": "1000",
                "chetime": "130430",
                "k200jisu": "430.40",
                "openyak": "100",
                "bidho1": "430.45",
                "offerho1": "430.55",
            },
        },
        {
            "ts_ms": 1000,
            "trcode": "FH0",
            "symbol": "A016XXXX",
            "tick": {
                "hotime": "130431",
                "totofferrem": 1500,
                "totbidrem": 1200,
                # compact schema: int(x100)
                "offerho1": 43055,
                "bidho1": 43045,
                **{f"offerrem{i}": 100 + i for i in range(1, 6)},
                **{f"bidrem{i}": 80 + i for i in range(1, 6)},
            },
        },
    ]

    with gzip.open(str(p), "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    predictor = PredictionPipeline(min_minute_bars_required=1, seq_len=5, use_llm=False)

    rc = run_replay_mode_with_predictor(str(p), predictor, speed=0.0, max_lines=None)
    assert int(rc) == 0

    # Sanity: the FH0 record should have been buffered.
    assert len(getattr(predictor, "_ob_records", []) or []) >= 1
