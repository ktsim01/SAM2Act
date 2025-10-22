#!/usr/bin/env bash
set -euo pipefail

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0

CLUSTER="1-18"
TASK="insert_onto_square_peg"

# if [[ $CLUSTER == "1-18" ]]; then
#     NP=8
#     BS=110
#     EPOCHS=600
#     EXP_PATH=/project_data/held/mnakuraf/tax3d-conditioned-mimicgen/third_party/robogen/test_PointNet2/exps
#     DATASET_PREFIX=/scratch/minon/
#     cd /project_data/held/mnakuraf/tax3d-conditioned-mimicgen/third_party/robogen/test_PointNet2
# elif [[ $CLUSTER == "0-25" ]]; then
#     NP=5
#     BS=50
#     EPOCHS=300
#     EXP_PATH=/project_data/held/mnakuraf/tax3d-conditioned-mimicgen/third_party/robogen/test_PointNet2/exps
#     DATASET_PREFIX=/scratch/minon/
#     cd /project_data/held/mnakuraf/tax3d-conditioned-mimicgen/third_party/robogen/test_PointNet2
# elif [[ $CLUSTER == "0-37" ]]; then
#     NP=8
#     BS=55
#     EPOCHS=300
#     EXP_PATH=/project_data/held/mnakuraf/tax3d-conditioned-mimicgen/third_party/robogen/test_PointNet2/exps
#     DATASET_PREFIX=/scratch/minon/
#     cd /project_data/held/mnakuraf/tax3d-conditioned-mimicgen/third_party/robogen/test_PointNet2
# elif [[ $CLUSTER == "atlas" ]]; then
#     NP=2
#     BS=55
#     EPOCHS=100
#     EXP_PATH=/data/minon/tax3d-conditioned-mimicgen/third_party/robogen/test_PointNet2/exps
#     SHORT_TASK=${TASK:0:-4}
#     DATASET_PREFIX=/data/minon/tax3d-conditioned-mimicgen/robomimic/datasets/${SHORT_TASK,,}
#     cd /data/minon/tax3d-conditioned-mimicgen/third_party/robogen/test_PointNet2
# else
#     echo "Unknown CLUSTER value: $CLUSTER"
#     exit 1
# fi

NP=1
BS=10
EPOCHS=500
EXP_PATH=/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps
DATASET_PREFIX=/home/ktsim/Projects/SAM2Act/sam2act/data_articubot/insert_onto_square_peg_10k

echo ${DATASET_PREFIX}/${TASK,,}/
echo _${TASK}
cd third_party/robogen/
torchrun --standalone --nproc_per_node=$NP val_eval_normal.py \
    --batch_size $BS \
    --num_epochs $EPOCHS \
    --model_type pointnet2_text_10k --model_invariant \
    --exp_path $EXP_PATH \
    --num_train_objects $TASK \
    --dataset_prefix ${DATASET_PREFIX} \
    --use_all_data \
    --use_text \
    --add_one_hot_encoding 1 \
    --use_color \
    --gmm \
    --load_model_path /home/ktsim/checkpoints/pointnet2_text_10k_model_invariant_2025-10-19_use_all_data_insert_onto_square_peg-obj_one_hot_use_color_use_text_gmm_/best_model.pth \
    # --load_model_path /home/ktsim/checkpoints/pointnet2_text_10k_model_invariant_2025-08-24_use_all_data_insert_onto_square_peg-obj_one_hot_use_color_use_text_/model_310.pth \
    # --load_model_path /home/ktsim/checkpoints/pointnet2_text_10k_model_invariant_2025-10-16_use_all_data_insert_onto_square_peg-obj_one_hot_use_color_use_text_gmm_/model_140.pth \
    # --load_model_path /home/ktsim/checkpoints/pointnet2_text_10k_model_invariant_2025-10-19_use_all_data_insert_onto_square_peg-obj_one_hot_use_color_use_text_gmm_/model_300.pth \
    # --wandb
