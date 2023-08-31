from datasets import load_dataset

# dataset_names = ["noisy", "noiseless", "noisy_16", "noisy_8", "noisy_4"]
dataset_names = ["DESY3", "LSSTY1", "LSSTY10"]

for dataset in dataset_names:
    print(dataset)
    # data = load_dataset("../data/20230419_224x224/20230419_224x224.py", dataset, cache_dir="/pscratch/sd/s/shubh/")
    data = load_dataset("../data/20230814_224x224/20230814_224x224.py", \
                        dataset, cache_dir="/pscratch/sd/s/shubh/")
    data = load_dataset("../data/20230814_224x224/20230814_224x224.py", \
                        dataset+"_onebin", cache_dir="/pscratch/sd/s/shubh/")
    data = load_dataset("../data/20230814_224x224/20230814_224x224.py", \
                        dataset+"_half", cache_dir="/pscratch/sd/s/shubh/")

data = load_dataset("../data/20230814_224x224/20230814_224x224.py", \
                        "DESY3_double", cache_dir="/pscratch/sd/s/shubh/")
