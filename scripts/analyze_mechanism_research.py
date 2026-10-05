from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from crypto_timing.mechanism_analysis import head_diagnostics, paired_skill_interval, signal_study
from crypto_timing.mechanism_training import MechanismStore, metrics

ROOT=Path("outputs/mechanism")


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    records=json.loads((ROOT/"screen_results.json").read_text())
    stages=json.loads((ROOT/"selection_log.json").read_text())
    confirmed=json.loads((ROOT/"confirmation_results.json").read_text())
    locked=json.loads((ROOT/"locked_design.json").read_text())
    uncertainty={}
    for stage in stages:
        first=stage["candidates"][0]
        a=np.load(ROOT/"screen"/records[first]["artifact_run"]/"validation.npz")
        for name in stage["candidates"][1:]:
            b=np.load(ROOT/"screen"/records[name]["artifact_run"]/"validation.npz")
            assert np.array_equal(a["hours"],b["hours"])
            uncertainty[f"{first} → {name}"]=paired_skill_interval(a["signal"],b["signal"],a["labels"])
    (ROOT/"paired_uncertainty.json").write_text(json.dumps(uncertainty,ensure_ascii=False,indent=2),encoding="utf-8")
    ensemble={}
    study={}
    native={}
    for fold in ("f1","f2"):
        main_folder=ROOT/"confirm"/(f"f1_{locked['selected_kind']}_seed20261004" if fold=="f1" else "f2_selected_seed20261004")
        folders=[main_folder]+[ROOT/"confirm"/f"{fold}_selected_seed{seed}" for seed in (20261005,20261006)]
        ensemble[fold]={}
        for split in ("validation","replay"):
            datasets=[np.load(p/f"{split}.npz") for p in folders]
            for data in datasets[1:]:
                assert np.array_equal(data["hours"],datasets[0]["hours"])
            pred=np.mean([data["signal"] for data in datasets],axis=0)
            data=datasets[0]
            ensemble[fold][split]=metrics(pred,data["labels"],data["dates"])
            ensemble[fold][split]["uncertainty_vs_zero"]=paired_skill_interval(np.zeros_like(pred),pred,data["labels"])
            np.savez_compressed(ROOT/f"ensemble_{fold}_{split}.npz",hours=data["hours"],dates=data["dates"],signal=pred,labels=data["labels"],raw=data["raw"],scale=data["scale"])
        study[fold]=signal_study(ROOT,locked["selected_config"],fold,main_folder)
        (ROOT/f"signal_study_{fold}.json").write_text(json.dumps(study[fold],ensure_ascii=False,indent=2),encoding="utf-8")
        torch.cuda.empty_cache()
        store=MechanismStore(ROOT/"cache_v4",fold,torch.device("cuda"))
        if fold=="f1":
            visited=set()
            for name,record in records.items():
                artifact=record["artifact_run"]
                if artifact not in visited:
                    native[name]=head_diagnostics(ROOT/"screen"/artifact,store)
                    visited.add(artifact)
            for name in ("return","joint","proposal"):
                native[f"confirm_f1_{name}"]=head_diagnostics(ROOT/"confirm"/f"f1_{name}_seed20261004",store)
        else:
            native["confirm_f2_selected"]=head_diagnostics(main_folder,store)
            native["confirm_f2_contrast"]=head_diagnostics(ROOT/"confirm"/"f2_label_contrast",store)
        del store
        torch.cuda.empty_cache()
    (ROOT/"ensemble_results.json").write_text(json.dumps(ensemble,indent=2),encoding="utf-8")
    (ROOT/"head_diagnostics.json").write_text(json.dumps(native,ensure_ascii=False,indent=2),encoding="utf-8")
    source_files=sorted(list(Path("src/crypto_timing").glob("mechanism_*.py"))+list(Path("scripts").glob("*mechanism*.py")))
    runtime={"python":sys.executable,"python_version":platform.python_version(),"torch":torch.__version__,
              "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),
              "numpy":np.__version__,"source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
              "screen_configurations":len(records),"unique_screen_training_runs":len({r["artifact_run"] for r in records.values()}),
              "confirmation_training_runs":len(confirmed),"selected":locked,
              "historical_status":"all evaluated history exploratory; no independently unseen forward data in current source"}
    (ROOT/"runtime_manifest.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"ensemble":ensemble,"filters":{f:s["learned_alpha"] for f,s in study.items()},"runtime":runtime},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
