. ~/miniconda3/etc/profile.d/conda.sh
conda activate lora
# 步骤2: 图片打标签
cd $ROOT_PATH/sd-scripts
python ./finetune/make_captions.py $TRAIN_DIR --batch_size=8 --caption_extension=".txt" --caption_weights=/root/autodl-tmp/lora-scripts/sd-scripts/model_large_caption.pth
cd $ROOT_PATH
# 步骤4: 加入关键词
python $ROOT_PATH/batch_keyword.py --dir_path=$TRAIN_DIR --keyword="$KEY_WORD,"
# 步骤5: 开始训练
bash -x $ROOT_PATH/train.sh
# LoRA train script by @Akegarasu