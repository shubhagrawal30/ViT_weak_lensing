from datasets import load_dataset
import os
from pathlib import Path

dataset_names = ["DESY3", "LSSTY1", "LSSTY10"]
addeds = ["", "_half", "_onebin", "_double", "_half_twice", "_twice", "_4times"]#, "_8times", "_16times", "_all"]
hashcode = "7e924e5aa3dc3eb2e0370985a404633e9a0e8c07c3d5d5028e07308b65991e43"

done = []
preds_done = []

for dataset in dataset_names:
    # if "LSSTY10" in dataset:
    #     cache_dir = "/pscratch/sd/h/helenqu/shubh/ViT/"
    # elif "DESY3" in dataset:
    #     cache_dir = "/pscratch/sd/m/mjarvis/shubh/"
    # else:
    cache_dir = "/pscratch/sd/s/shubh/ViT/"
    for added in addeds:
        if os.path.exists(f"/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/new/temp/RN_{dataset}{added}.lock") or \
          os.path.exists(f"/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/new/temp/RN_{dataset}{added}.done") or \
          not os.path.exists(f"{cache_dir}/20230814_224x224/{dataset}{added}/1.1.0/{hashcode}/"):
            print(f"Skipping {dataset}{added}")
            continue
        if f"{dataset}{added}" in done:
            print(f"Already done {dataset}{added}")
            if f"{dataset}{added}" in preds_done:
                continue
            os.system(f"cd /global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/; conda activate vit; python resnet.py {dataset}{added} 0;")
            os.system(f"cd /global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/; conda activate vit; python resnet.py {dataset}{added} 1;")
            os.rmdir(f"/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/new/temp/RN_{dataset}{added}.lock")
        else:
            Path(f"/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/new/temp/RN_{dataset}{added}.lock").mkdir(parents=True, exist_ok=True)
            print(f"Running ResNet on {dataset}{added}")
            os.system(f"cd /global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/; conda activate vit; python resnet.py {dataset}{added} 0;")
            # Path(f"/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/new/temp/RN_{dataset}{added}.done").mkdir(parents=True, exist_ok=True)
            os.rmdir(f"/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/new/temp/RN_{dataset}{added}.lock")
        # break