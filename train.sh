set -ex
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot /home/data/MI3DOR_Spilt/Spilt1  --name workshop_baseline_notexture_tuning_v1 --model retrieval_workshop_baseline_tuning --dataset_mode retrieval_workshop_baseline --niter 30 --niter_decay 70 --crop_size 256 --fine_size 256 --num_threads 8 --lr 0.001 --batch_size 16  --gpu_ids 0 --continue_train --drop --epoch_count 47 

