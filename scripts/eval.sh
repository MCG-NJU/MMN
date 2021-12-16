# find all configs in configs/
#config_file=configs/pool_tacos_128x128_k5l8.yaml
config_file=configs/pool_activitynet_64x64_k9l4.yaml
#config_file=configs/pool_charades_16x16_k5l8.yaml
# the dir of the saved weight
weight_dir=outputs/pool_activitynet_64x64_k9l4
#weight_dir=outputs/pool_charades_16x16_k5l8
#weight_dir=outputs/pool_tacos_128x128_k5l8/
# select weight to evaluate
#weight_file=outputs/pool_charades_16x16_k5l8/best_charades.pth
#weight_file=outputs/pool_tacos_128x128_k5l8/best_tacos.pth
weight_file=outputs/pool_activitynet_64x64_k9l4/best_anet.pth
# test batch size
batch_size=16
# set your gpu id
gpus=0
# number of gpus
gpun=1
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi mmn task on the same machine
master_addr=127.0.0.2
master_port=29578

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port \
test_net.py --config-file $config_file --ckpt $weight_file OUTPUT_DIR $weight_dir TEST.BATCH_SIZE $batch_size

