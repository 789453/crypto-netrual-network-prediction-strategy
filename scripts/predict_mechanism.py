"""Verify inference at the newest completed hour, whose future labels do not exist."""
import argparse
import json
from pathlib import Path
import numpy as np
from crypto_timing.mechanism_inference import MechanismPredictor

parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint",type=Path,required=True)
parser.add_argument("--cache",type=Path,default=Path("outputs/mechanism/cache_v4"))
args=parser.parse_args()
predictor=MechanismPredictor(args.checkpoint)
manifest=json.loads((args.cache/"manifest.json").read_text())
dates=np.load(args.cache/"decision_time.npy")
h=len(dates)-1
symbols=np.arange(len(manifest["symbols"]))
inputs={}
for name,multiple,window in (("fast",12,144),("slow",1,predictor.cfg.slow_hours)):
    arr=np.load(args.cache/f"{name}_{predictor.cfg.prep}.npy",mmap_mode="r")
    end=(h+1)*multiple
    inputs[name]=np.asarray(arr[end-window:end]).transpose(1,0,2)
inputs["state"]=np.load(args.cache/f"state_{predictor.cfg.prep}.npy",mmap_mode="r")[h]
if predictor.cfg.minute in ("encoder","aggregate"):
    name,multiple,window=("minute",60,180) if predictor.cfg.minute=="encoder" else ("extra",12,144)
    arr=np.load(args.cache/f"{name}.npy",mmap_mode="r")
    end=(h+1)*multiple
    inputs[name]=np.asarray(arr[end-window:end]).transpose(1,0,2)
# This column is past-only risk and is populated even when the future-return columns are NaN.
raw=np.load(args.cache/"targets.npy",mmap_mode="r")
assert np.isnan(raw[h,:,:3]).all(), "expected unavailable future labels at the newest completed hour"
out=predictor.predict(inputs,symbols,np.asarray(raw[h,:,7]))
result={"decision_utc":str(dates[h]),"future_labels_available":False,"symbols":manifest["symbols"],
        **{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in out.items()}}
Path("outputs/mechanism/latest_inference.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False),flush=True)
