"""Restore the v3 zero-return direction convention in the diagnostic four-bin arm."""
import json
from pathlib import Path
import torch
from dataclasses import replace
from crypto_timing.mechanism_model import MechanismConfig
from crypto_timing.mechanism_training import MechanismStore, train_trial

root=Path("outputs/mechanism")
records=json.loads((root/"screen_results.json").read_text())
if records["label_four_bin"].get("zero_return_policy")!="half_direction" or records["label_four_bin"].get("prior_initialization_policy")!="train_bin_priors":
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    store=MechanismStore(root/"cache_v4","f1",torch.device("cuda"))
    cfg=MechanismConfig(**records["label_four_bin"]["config"])
    corrected=train_trial(store,cfg,root/"screen"/"label_four_bin_prior_corrected")
    corrected.update({"artifact_run":"label_four_bin_prior_corrected","zero_return_policy":"half_direction","prior_initialization_policy":"train_bin_priors"})
    records["label_four_bin"]=corrected
    (root/"screen_results.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    stages=json.loads((root/"selection_log.json").read_text())
    for stage in stages:
        if stage["stage"]=="label":
            stage["selected"]=min(stage["candidates"],key=lambda name:records[name]["results"]["validation"]["mse"])
            stage["note"]="four-bin exact-zero direction restored to 0.5 and retrained; label stage diagnostic, P+R arm retained for architecture questions"
    (root/"selection_log.json").write_text(json.dumps(stages,ensure_ascii=False,indent=2),encoding="utf-8")
    print("four-bin diagnostic corrected and retrained",flush=True)
if "horizon_4" not in records:
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    store=MechanismStore(root/"cache_v4","f1",torch.device("cuda"))
    cfg=replace(MechanismConfig(**records["horizon_1"]["config"]),horizon=4)
    matched=train_trial(store,cfg,root/"screen"/"horizon_4")
    matched["artifact_run"]="horizon_4"
    records["horizon_4"]=matched
    (root/"screen_results.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    print("matched-budget 4h horizon arm completed",flush=True)
