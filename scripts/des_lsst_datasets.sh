script='../data/20230814_224x224/20230814_224x224.py'
cache='/pscratch/sd/s/shubh/'

for inst in "DESY3" "LSSTY1" "LSSTY10"
do
    python -c "from datasets import load_dataset; \
        load_dataset('$script', '$inst'+'_onebin', cache_dir='$cache')" &

    python -c "from datasets import load_dataset; \
        load_dataset('$script', '$inst'+'_half', cache_dir='$cache')" &

    python -c "from datasets import load_dataset; \
        load_dataset('$script', '$inst', cache_dir='$cache')" &
done

python -c "from datasets import load_dataset; \
    load_dataset('$script', 'DESY3_double', cache_dir='$cache')" &