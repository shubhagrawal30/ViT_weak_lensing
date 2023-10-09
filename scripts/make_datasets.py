from datasets import load_dataset
import os

dataset_names = ["DESY3", "LSSTY1", "LSSTY10"]
addeds = ["", "_half", "_onebin", "_double", "_half_twice", "_twice", "_4times"]#, "_8times", "_16times", "_all"]
# addeds = ["_onebin"]
# hashcode = "fdbec89db2168787deac1ef744190b64434299a1eed5f8e7adeb610438d0c926"
hashcode = "7e924e5aa3dc3eb2e0370985a404633e9a0e8c07c3d5d5028e07308b65991e43"

# os.path.exists(f"{cache_dir}/20230814_224x224/{dataset}{added}/1.1.0/{hashcode}.incomplete") or \

for dataset in dataset_names:
    # if "LSSTY10" in dataset:
    #     cache_dir = "/pscratch/sd/h/helenqu/shubh/ViT/"
    # elif "DESY3" in dataset:
    #     cache_dir = "/pscratch/sd/m/mjarvis/shubh/"
    # else:
    cache_dir = "/pscratch/sd/s/shubh/ViT/"
    for added in addeds:
        if os.path.exists(f"{cache_dir}/20230814_224x224/{dataset}{added}/1.1.0/{hashcode}_builder.lock") or \
          os.path.exists(f"{cache_dir}/20230814_224x224/{dataset}{added}/1.1.0/{hashcode}/"):
            print(f"Skipping {dataset}{added}")
            continue
        print(f"Downloading {dataset}{added}")
        load_dataset("../data/20230814_224x224/20230814_224x224.py", \
                            dataset + added, cache_dir=cache_dir)
        # break