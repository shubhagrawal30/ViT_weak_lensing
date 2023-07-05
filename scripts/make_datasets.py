from datasets import load_dataset

dataset_names = ["noisy", "noiseless", "noisy_16", "noisy_8", "noisy_4"][2:]

for dataset in dataset_names:
    print(dataset)
    data = load_dataset("../data/20230419_224x224/20230419_224x224.py", dataset, cache_dir="/pscratch/sd/s/shubh/")
