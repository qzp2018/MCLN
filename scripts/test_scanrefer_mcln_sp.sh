TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node 4 --master_port 4444 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ~/DATA_ROOT/ \
    --val_freq 1 --batch_size 6 --save_freq 1 --print_freq 500 \
    --lr_backbone=2e-3 --lr=2e-4 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ~/DATA_ROOT/output/logs/ \
    --lr_decay_epochs 50 75 \
    --butd --self_attend --augment_det \
    --checkpoint_path ~/DATA_ROOT/checkpoints/ckpt.pth \
    --max_epoch 100 \
    --model MCLN \
    --small_lr \
    --exp MCLN \
    --eval \