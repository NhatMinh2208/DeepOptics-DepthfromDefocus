CUDA_VISIBLE_DEVICES=2 python3 minitrain.py --batch_sz 3  --num_workers 8   --max_epochs 100  --optimize_optics  --psfjitter 
#python3 train.py --batch_sz 3  --num_workers 8   --max_epochs 20  --optimize_optics  --psfjitter 
#CUDA_VISIBLE_DEVICES=2 python3 smalltrain.py --batch_sz 3  --num_workers 8   --max_epochs 40  --optimize_optics  --psfjitter 
#python snapshotdepth_trainer.py  --gpus 4 --batch_sz 3  --num_workers 0 --distributed_backend ddp  --max_epochs 100  --optimize_optics  --psfjitter  --replace_sampler_ddp False