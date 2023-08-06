# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_set
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/7/24 13:57
"""
    文件说明：
            
"""
import json
import torch
from torch.utils.data import Dataset


class GLMPromptDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, is_skip):
        self.all_data = []
        self.tokenizer = tokenizer
        skip_data_number = 0
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                skip_flag = False
                src_tokens = tokenizer.tokenize(sample["instruction"] + sample["input"])

                if len(src_tokens) > max_src_len:
                    src_tokens = src_tokens[:max_src_len]
                    skip_flag = True

                max_tgt_len = max_len - 3 - len(src_tokens)
                tgt_tokens = tokenizer.tokenize(sample["output"])

                if len(tgt_tokens) > max_tgt_len:
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                    skip_flag = True

                tokens = src_tokens + ["[gMASK]", "<sop>"] + tgt_tokens + ["<eop>"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1:]

                assert len(input_ids) == len(labels)
                assert len(input_ids) <= max_len
                if is_skip and skip_flag:
                    skip_data_number += 1
                    continue
                self.all_data.append({"input_ids": input_ids, "labels": labels})
        print("the number of skipping data is {}".format(skip_data_number))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance

    def __call__(self, instances):
        input_ids, labels = tuple([torch.tensor(instance[key], dtype=torch.long) for instance in instances] for key in
                                  ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels
        )


class GLM2PromptDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, is_skip):
        self.all_data = []
        self.tokenizer = tokenizer
        skip_data_number = 0
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                skip_flag = False
                src_tokens = tokenizer.tokenize(sample["instruction"] + sample["input"])

                if len(src_tokens) > max_src_len:
                    src_tokens = src_tokens[:max_src_len]
                    skip_flag = True

                max_tgt_len = max_len - 3 - len(src_tokens)
                tgt_tokens = tokenizer.tokenize(sample["output"])

                if len(tgt_tokens) > max_tgt_len:
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                    skip_flag = True

                input_ids = [64790, 64792] + input_ids
                labels = [-100, -100] + labels
                if len(input_ids) > max_len:
                    skip_data_number += 1
                    continue

                tokens = src_tokens + tgt_tokens + ["</s>"]
                assert len(tokens) <= max_len
                input_ids = [64790, 64792] + tokenizer.convert_tokens_to_ids(tokens)
                context_length = len(src_tokens) + 2
                labels = [-100] * context_length + input_ids[context_length:]
                assert len(input_ids) == len(labels)
                if is_skip and skip_flag:
                    skip_data_number += 1
                    continue
        print("the number of skipping data is {}".format(skip_data_number))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance

    def __call__(self, instances):
        input_ids, labels = tuple([torch.tensor(instance[key], dtype=torch.long) for instance in instances] for key in
                                  ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels
        )

# def collate_fn_glm2(batch_data):
#     batch_size = len(batch_data)
#     if batch_size == 0:
#         return {}
#     input_ids_list, attention_mask_list, label_list = [], [], []
#     for instance in batch_data:
#         input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
#         label_list.append(torch.tensor(instance["labels"], dtype=torch.long))
#     return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
#             "labels": pad_sequence(label_list, batch_first=True, padding_value=-100)}
