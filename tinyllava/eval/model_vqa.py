import argparse
import torch
import os
import json
import time
from tqdm import tqdm
import shortuuid

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    model.to(device='cuda')    
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    if "lingoqa" in args.question_file.lower():
        with open(args.question_file, 'r', encoding='utf-8') as file:  
            data = json.load(file)
        questions = [
            {
                "question_id": example["id"],
                "image_path_list": example["image"],
                "text": example["conversations"][0]["value"].replace("<image>\n", ""),
                **({"source_id": example["source_id"], "time_length": example["time_length"]} if "bdd" in args.question_file.lower() else {})
            }
            for example in data
        ]
    else:
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    if "lingoqa" in args.question_file.lower():
        questions = questions[::2]

    # Performance metrics initialization
    total_frames = 0
    torch.cuda.reset_peak_memory_stats(device="cuda:0")
    start_time = time.time()
        
    for line in tqdm(questions):
        idx = line["question_id"]
        image_path_list = line["image_path_list"]
        qs = line["text"]
        
        cur_prompt = qs
        
        question_ids = tokenizer(cur_prompt,
                        return_tensors="pt",
                        padding='max_length',
                        max_length=20,
                        truncation=True).input_ids
        
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        msg = Message()
        msg.add_message(qs)

        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        prompt = result['prompt']
        input_ids = input_ids.unsqueeze(0).cuda()

        image_tensors = []
        for img_path in image_path_list:
            image = Image.open(os.path.join(args.image_folder, img_path)).convert('RGB')
            image_tensor = image_processor(image)
            image_tensors.append(image_tensor)
        
        image_tensors = torch.stack(image_tensors, dim=0)
        
        image_tensors = image_tensors.unsqueeze(0).half().cuda()
        image_sizes = [image.size]
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
                question_ids=question_ids.cuda())

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        frame_end_time = time.time()
        total_frames += 1
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": args.model_base,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    
    end_time = time.time()
    total_duration = end_time - start_time
    fps = total_frames / total_duration
    peak_memory_used = torch.cuda.max_memory_allocated(device="cuda:0") / (1024 ** 3)  # GB

    ans_file.close()
    print(f"Total frames processed: {total_frames}")
    print(f"Total time: {total_duration:.2f} seconds")
    print(f"FPS (Frames Per Second): {fps:.2f}")
    print(f"Peak GPU memory used: {peak_memory_used:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen2_base")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
