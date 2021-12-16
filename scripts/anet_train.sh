# find all configs in configs/
config=pool_activitynet_64x64_k9l4
# set your gpu id
gpus=0,1,2,3,4,5
# number of gpus
gpun=6
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi mmn task on the same machine
master_addr=127.0.0.3
master_port=29511

# ------------------------ need not change -----------------------------------
config_file=configs/$config\.yaml
output_dir=outputs/$config

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port train_net.py --config-file $config_file OUTPUT_DIR $output_dir \

