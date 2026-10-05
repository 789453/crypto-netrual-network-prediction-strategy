import json
from pathlib import Path
from crypto_timing.mechanism_analysis import feature_audit

root = Path("outputs/mechanism")
for fold in ("f1","f2"):
    report = feature_audit(root / "cache_v4",fold)
    (root / f"feature_audit_{fold}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(fold,report["sample_rows"],report["near_pairs"][:4],flush=True)
