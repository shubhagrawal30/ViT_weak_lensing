PYTHON_FILE="run_ViT.py"
# PYTHON_FILE="run_ResNet.py"
# PYTHON_FILE="make_datasets.py"

for i in {11..50}
do
    echo $i
    ssh login$i -tt "cd $CDIRS/transformers/ViT_weak_lensing/scripts; conda activate vit; python $PYTHON_FILE;" &
    sleep 10
done
