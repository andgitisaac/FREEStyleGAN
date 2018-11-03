CUDA_VISIBLE_DEVICES=0 python -u finetune_decoder.py --save_dir ./checkpoints/finetune_decoder --log_dir 
./logs/finetune_decoder --pretrained_decoder ./pretrained/pretrained_decoder.pth --batch_size 4
