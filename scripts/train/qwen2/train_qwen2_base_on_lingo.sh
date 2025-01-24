FINETUNE_DATA_PATH=playground/data/LingoQA/train.json #finetune annotation file path
FINETUNE_IMAGE_PATH=""
LLM_VERSION=/mnt/csi-data-aly/user/haozhou/Projects/TinyLLaVA_Factory/checkpoints/Qwen/Qwen2-0.5B # llm path in huggingface
VT_VERSION=/mnt/csi-data-aly/shared/public/haozhou/checkpoints/siglip/siglip-so400m-patch14-384/ #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=qwen2_base #chat template, other options are: phi, llama, gemmma, etc
VERSION=qwen2-0_5b_base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=8192 #max model length for llm

bash scripts/train/qwen2/finetune_qwen2_on_lingo.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
