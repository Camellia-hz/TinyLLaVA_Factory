
FINETUNE_DATA_PATH=playground/data/LingoQA/train.json
FINETUNE_IMAGE_PATH=""
LLM_VERSION=/mnt/csi-data-aly/user/haozhou/Projects/TinyLLaVA_Factory/checkpoints/google/gemma-2b-it
VT_VERSION=/mnt/csi-data-aly/shared/public/haozhou/checkpoints/siglip/siglip-so400m-patch14-384/
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
CONV_VERSION=gemma
VERSION=base
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=8192

bash scripts/train/gemma/finetune_gemma_lingoqa.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
