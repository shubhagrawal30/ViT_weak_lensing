# CUDA_VISIBLE_DEVICES=5 python ViT.py 0 &
# CUDA_VISIBLE_DEVICES=4 python ViT.py 1 &
CUDA_VISIBLE_DEVICES=2 python ViT.py 2 &
# CUDA_VISIBLE_DEVICES=5 python resnet.py 0 &
CUDA_VISIBLE_DEVICES=4 python resnet.py 1 &
CUDA_VISIBLE_DEVICES=3 python resnet.py 2 &
