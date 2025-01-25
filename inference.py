import sys
sys.path.append('.')

from tinyllava.eval.run_tiny_llava import eval_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from tinyllava_visualizer.tinyllava_visualizer import *
from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

# model = AutoModelForCausalLM.from_pretrained("/mnt/csi-data-aly/user/haozhou/Projects/TinyLLaVA_Factory/checkpoints/tinyllava/TinyLLaVA-Qwen2-0.5B-SigLIP", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("/mnt/csi-data-aly/user/haozhou/Projects/TinyLLaVA_Factory/checkpoints/tinyllava/TinyLLaVA-Qwen2-0.5B-SigLIP", trust_remote_code=True)
# model.tokenizer = tokenizer
model_path = "/mnt/csi-data-aly/user/haozhou/Projects/TinyLLaVA_Factory/checkpoints/tinyllava/TinyLLaVA-Qwen2-0.5B-SigLIP"
model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)

args = type('Args', (), {
    "model_path": None,
    "model": model,
    "query": prompt,
    "conv_mode": "qwen2_base", # the same as conv_version in the training stage. Different LLMs have different conv_mode/conv_version, please replace it
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

monitor = Monitor(args, model, llm_layers_index=23)
eval_model(args)
monitor.get_output(output_dir='results/')