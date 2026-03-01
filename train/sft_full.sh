CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 ../LLaMA-Factory/src/train.py \
   ../train/qwen3vl_8b_full_sft.yaml  

