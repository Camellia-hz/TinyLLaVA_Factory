import copy
from dataclasses import dataclass
import json
from typing import Dict,  Sequence, TYPE_CHECKING
from PIL import Image, ImageFile
import os

from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *


import transformers
import torch
from torch.utils.data import Dataset

def preprocess_question(sources,
                        tokenizer,
                        data_dict: dict,
                        max_length: int=20):
    
    human_conversation = next(item['value'] for item in sources['conversations'] if item['from'] == 'human')

    question = human_conversation.replace('<image>', '').strip()
    
    question_ids = tokenizer(question,
                        return_tensors="pt",
                        padding='max_length',
                        max_length=max_length,
                        truncation=True).input_ids
  
    data_dict['question_ids'] = question_ids.squeeze(0)
    
    return data_dict

ImageFile.LOAD_TRUNCATED_IMAGES = True

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)
        self.max_length = 20

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))
        if 'image' in sources:
            image_file = sources['image']
            image_folder = self.data_args.image_folder
            
            if isinstance(image_file, list):
                images = [Image.open(os.path.join(image_folder, img_file)).convert('RGB') for img_file in image_file]
                images = [self.image_preprocess(img) for img in images]
                data_dict['images'] = images
            else:
                image = Image.open(os.path.join(image_folder, images)).convert('RGB')
                image = self.image_preprocess(image)
                data_dict['images'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            print("xxxxxxxx")
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['images'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            
        data_dict = preprocess_question(sources, self.tokenizer, data_dict, max_length=self.max_length)
        
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, labels, question_ids = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "question_ids"))
        
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            question_ids=torch.stack(question_ids, dim=0),
        )

        if 'images' in instances[0]:
            images = [instance['images'] for instance in instances]
            if isinstance(images[0], list):
                images = torch.stack([torch.stack(img, dim=0) for img in images], dim = 0)
                batch['images'] = images
            else:
                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch['images'] = torch.stack(images)
                else:
                    batch['images'] = images
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
