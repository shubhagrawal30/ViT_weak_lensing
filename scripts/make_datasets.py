from datasets import load_dataset
import os

dataset_type = "20231010_224x224_2par"
dataset_names = ["DESY3", "LSSTY1", "LSSTY10"]
addeds = ["", "_half", "_onebin", "_double", "_half_twice", "_twice", "_4times", "_8times", "_16times", \
          "_32times", "_64times", "_128times", "_all"]
# addeds = ["_onebin"]
hashcode = "5cc2341c4f63ee4d85bc4f0695f06015c0f88489290ed7f1fab5228186aa62fe"

# os.path.exists(f"{cache_dir}/{dataset_type}/{dataset}{added}/1.1.0/{hashcode}.incomplete") or \

for dataset in dataset_names:
    cache_dir = "/pscratch/sd/s/shubh/ViT/"
    for added in addeds:
        if os.path.exists(f"{cache_dir}/{dataset_type}/{dataset}{added}/1.1.0/{hashcode}_builder.lock") or \
          os.path.exists(f"{cache_dir}/{dataset_type}/{dataset}{added}/1.1.0/{hashcode}/"):
            print(f"Skipping {dataset}{added}")
            continue
        print(f"Downloading {dataset}{added}")
        load_dataset(f"../data/{dataset_type}/{dataset_type}.py", \
                            dataset + added, cache_dir=cache_dir)
        # break