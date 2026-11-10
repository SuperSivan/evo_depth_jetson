# Three-stage EvoDepth training (example: LIBERO-style config).
# Stage 1: train action head only (--finetune_action_head). DA3 may run in forward with --use_da3 but stays frozen without --finetune_da3.
# Stage 2: train DA3 + action head (--finetune_da3 --finetune_action_head), resume from stage 1.
# Stage 3: full fine-tune VLM + DA3 + action head (--finetune_vlm --finetune_da3 --finetune_action_head), resume from stage 2.
#
# Edit --dataset_config_path, --save_dir, --resume_path, and step tags to match your runs.

# ========== Stage 1 ==========
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --deepspeed_config_file ds_config.json \
  --main_process_port 29519 \
  scripts/train.py \
  --run_name Evo_depth_3stages_stage1 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 16 \
  --image_size 448 \
  --max_steps 5000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --finetune_action_head \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path dataset/config.yaml \
  --per_action_dim 24 \
  --state_dim 24 \
  --use_da3 \
  --save_dir /path/to/checkpoints/stage1/

# ========== Stage 2 ==========
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --deepspeed_config_file ds_config.json \
  --main_process_port 29519 \
  scripts/train.py \
  --run_name Evo_depth_3stages_stage2 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 16 \
  --image_size 448 \
  --max_steps 10000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --finetune_action_head \
  --finetune_da3 \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path dataset/config.yaml \
  --per_action_dim 24 \
  --state_dim 24 \
  --use_da3 \
  --save_dir /path/to/checkpoints/stage2/ \
  --resume \
  --resume_pretrain \
  --resume_path /path/to/checkpoints/stage1/step_5000

# ========== Stage 3 ==========
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --deepspeed_config_file ds_config.json \
  --main_process_port 29519 \
  scripts/train.py \
  --run_name Evo_depth_3stages_stage3 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 16 \
  --image_size 448 \
  --max_steps 80000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --use_da3 \
  --finetune_vlm \
  --finetune_action_head \
  --finetune_da3 \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path dataset/config.yaml \
  --per_action_dim 24 \
  --state_dim 24 \
  --save_dir /path/to/checkpoints/stage3/ \
  --resume \
  --resume_pretrain \
  --resume_path /path/to/checkpoints/stage2/step_10000
