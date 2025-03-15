#!/bin/bash

data_dir=data
image_size=224
patch_size=16
color_channels=3
hidden_size=128
num_heads=4
num_layers=4
norm_epsilon=1e-4
dropout=0.1
batch_size=32
epochs=100
learning_rate=1e-4
weight_decay=0
gradient_accumulation_steps=1
gradient_clipping=1.0
precision="fp16"
device="cuda"
log_and_eval_step=100
save_steps=100
num_workers=1
seed=42
save_directory="my_checkpoint"


#Get classes from data directory
cd $data_dir || exit
classes=$(ls -d */ | wc -l)  # Count number of directories (each representing a class)
cd ..
echo "Number of classes: $classes"
python pipeline.py \
    --data_dir $data_dir \
    --image_size $image_size \
    --patch_size $patch_size \
    --color_channels $color_channels \
    --hidden_size $hidden_size \
    --num_heads $num_heads \
    --num_layers $num_layers \
    --norm_epsilon $norm_epsilon \
    --dropout $dropout \
    --num_class $classes \
    --batch_size $batch_size \
    --epochs $epochs \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --gradient_clipping $gradient_clipping \
    --precision $precision \
    --device $device \
    --log_and_eval_step $log_and_eval_step \
    --save_steps $save_steps \
    --num_workers $num_workers \
    --save_directory $save_directory \
    --seed $seed \
    --pin_memory \
    --shuffle_data \
    --split_data \
    --normalize \
    --do_predict
