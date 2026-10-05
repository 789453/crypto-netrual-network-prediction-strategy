"""Sequential hypothesis ablations followed by full-clock, multi-seed forward replay."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from crypto_timing.mechanism_model import MechanismConfig
from crypto_timing.mechanism_training import MechanismStore, train_trial

ROOT = Path("outputs/mechanism")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA is mandatory; no CPU fallback"
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    store = MechanismStore(ROOT / "cache_v4", "f1", device)
    if args.smoke:
        cfg = MechanismConfig(minute="encoder")
        train_trial(store, cfg, ROOT / "smoke", epochs=2, steps_per_epoch=4)
        return
    records, stages = {}, []

    def trial(name, cfg):
        for previous_name, previous in records.items():
            if previous["config"] == cfg.__dict__:
                summary = {**previous, "artifact_run": previous.get("artifact_run", previous_name)}
                records[name] = summary
                (ROOT / "screen_results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
                return summary
        # Never reuse the archived four-bin run with the previous zero/prior policy.
        artifact = "label_four_bin_prior_corrected" if cfg.label == "four_bin" else name
        summary = train_trial(store, cfg, ROOT / "screen" / artifact)
        summary["artifact_run"] = artifact
        if cfg.label == "four_bin":
            summary.update(zero_return_policy="half_direction", prior_initialization_policy="train_bin_priors")
        records[name] = summary
        (ROOT / "screen_results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def compare(stage, candidates):
        scored = [(name, cfg, trial(name, cfg)) for name, cfg in candidates]
        winner = min(scored, key=lambda v: v[2]["results"]["validation"]["mse"])
        stages.append({"stage": stage, "candidates": [n for n, _, _ in scored], "selected": winner[0],
                       "selection": "f1 direct signal validation MSE; all history exploratory"})
        (ROOT / "selection_log.json").write_text(json.dumps(stages, ensure_ascii=False, indent=2), encoding="utf-8")
        return winner[1]

    base = MechanismConfig(label="return", memory="mean", features="all", fast_dim=19, modulation=False)
    base = compare("preprocessing", [(f"prep_{p}", replace(base, prep=p)) for p in ("raw", "field", "dynamic")])
    base = compare("redundancy", [(f"features_{f}", replace(base, features=f, fast_dim=19 if f == "all" else 16 if f == "original" else 14))
                                  for f in ("original", "all", "selected", "random")])
    # Keep both direct-return and path-auxiliary routes; compare labels without dismissing deep learning.
    labelled = [(f"label_{label}", replace(base, label=label))
                for label in ("return", "path", "joint", "distribution", "four_bin")]
    compare("label", labelled)
    joint = replace(base, label="joint")
    joint = compare("sharing", [(f"sharing_{s}", replace(joint, sharing=s)) for s in ("isolated", "partial", "full")])
    joint = compare("memory", [(f"memory_{m}", replace(joint, memory=m)) for m in ("mean", "ordered", "gru", "lstm")])
    joint = compare("width", [(f"width_{w}", replace(joint, width=w)) for w in (64,96,128)])
    joint = compare("state_modulation", [(f"modulation_{v}", replace(joint, modulation=v)) for v in (False,True)])
    joint = compare("slow_background", [(f"slow_{h}", replace(joint, slow_hours=h)) for h in (72,168)])
    joint = compare("minute_information", [(f"minute_{m}", replace(joint, minute=m)) for m in ("none","aggregate","encoder")])
    trial("path_sampling_1m", replace(joint, path_frequency=1))
    for h in (1,4,8):
        trial(f"horizon_{h}", replace(joint, label="return", horizon=h))
    # A complete proposal with all requested mechanisms remains inspectable even when simpler ablations win.
    proposal = replace(joint, memory="gru", sharing="partial", minute="encoder", modulation=True, slow_hours=168, width=96)
    trial("full_document_proposal", proposal)
    # Full-clock finalists get equal budgets, not just unequal screen exposure.
    final = {}
    full_return = replace(joint, label="return")
    for name, cfg in (("return", full_return), ("joint", joint), ("proposal", proposal)):
        final[f"f1_{name}"] = train_trial(store, cfg, ROOT / "confirm" / f"f1_{name}_seed20261004",
                                         epochs=5, full=True)
    best_kind = min(("return", "joint", "proposal"), key=lambda k: final[f"f1_{k}"]["results"]["validation"]["mse"])
    chosen = {"return": full_return, "joint": joint, "proposal": proposal}[best_kind]
    locked = {"selected_kind": best_kind, "selected_config": chosen.__dict__, "proposal_config": proposal.__dict__,
              "basis": "full-clock f1 validation MSE, before inspecting f2", "screen_exploratory": True}
    (ROOT / "locked_design.json").write_text(json.dumps(locked, indent=2), encoding="utf-8")
    for seed in (20261005,20261006):
        cfg = replace(chosen, seed=seed)
        final[f"f1_selected_{seed}"] = train_trial(store, cfg, ROOT / "confirm" / f"f1_selected_seed{seed}", epochs=5, full=True)
    (ROOT / "preprocessing_audit_f1.json").write_text(json.dumps(store.prep_audit, indent=2), encoding="utf-8")
    del store
    torch.cuda.empty_cache()
    store = MechanismStore(ROOT / "cache_v4", "f2", device)
    for seed in (20261004,20261005,20261006):
        cfg = replace(chosen, seed=seed)
        final[f"f2_selected_{seed}"] = train_trial(store, cfg, ROOT / "confirm" / f"f2_selected_seed{seed}", epochs=5, full=True)
    # Matched R/P+R forward contrast: train-only update, not a fresh parameter search on f2.
    contrast = replace(chosen, label="joint" if chosen.label == "return" else "return")
    final["f2_label_contrast"] = train_trial(store, contrast, ROOT / "confirm" / "f2_label_contrast", epochs=5, full=True)
    (ROOT / "preprocessing_audit_f2.json").write_text(json.dumps(store.prep_audit, indent=2), encoding="utf-8")
    (ROOT / "confirmation_results.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "research_complete", "screen_runs": len(records), "confirmation_runs": len(final),
                      "selected": locked}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
