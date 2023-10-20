from datasets import load_dataset
import os
from pathlib import Path

model_date = "20231010_VIT_"
dataset_type = "20231010_224x224_2par"
# dataset_names = ["DESY3", "LSSTY1", "LSSTY10"]
# addeds = ["", "_half", "_onebin", "_double", "_half_twice", "_twice", "_4times", "_8times", "_16times", \
#           "_32times", "_64times", "_128times", "_all"]
dataset_names = ["LSSTY1"]
addeds = ["16times", "32times", "8times", "double"]
hashcode = "5cc2341c4f63ee4d85bc4f0695f06015c0f88489290ed7f1fab5228186aa62fe"

done = []
preds_done = []

for dataset in dataset_names:
    cache_dir = "/pscratch/sd/s/shubh/ViT/"
    for added in addeds:
        if os.path.exists(f"/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/2par/temp/{dataset}{added}.lock") or \
          os.path.exists(f"/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/2par/temp/{dataset}{added}.done") or \
          not os.path.exists(f"{cache_dir}/{dataset_type}/{dataset}{added}/1.1.0/{hashcode}/") or \
          os.path.exists(f"/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/2par/models/{model_date}{dataset}{added}/preds0.npy"):
            print(f"Skipping {dataset}{added}")
            continue
        if f"{dataset}{added}" in done:
            print(f"Already done {dataset}{added}")
            if f"{dataset}{added}" in preds_done:
                print(f"preds also already done {dataset}{added}")
                continue
            Path(f"/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/2par/temp/{dataset}{added}.lock").mkdir(parents=True, exist_ok=True)
            os.system(f"cd /global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/; conda activate vit; python ViT.py {dataset}{added} 1;")
            os.rmdir(f"/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/2par/temp/{dataset}{added}.lock")
            break
        else:
            Path(f"/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/2par/temp/{dataset}{added}.lock").mkdir(parents=True, exist_ok=True)
            print(f"Running ViT on {dataset}{added}")
            os.system(f"cd /global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/; conda activate vit; python ViT.py {dataset}{added} 0;")
            # Path(f"/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/2par/temp/{dataset}{added}.done").mkdir(parents=True, exist_ok=True)
            os.rmdir(f"/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/2par/temp/{dataset}{added}.lock")
        # break