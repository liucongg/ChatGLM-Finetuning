# -*- coding:utf-8 -*-
# @project: ChatGLM-Finetuning
# @filename: convert_to_hf
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/6/5 11:06
"""
    文件说明：
            
"""
import torch
from pathlib import Path
import os
from os.path import join
from shutil import copy
import argparse


def convert_model_to_hf(ori_model_dir, pipeline_model_dir, save_model_dir):
    model_static_dict = {}
    for path in Path(pipeline_model_dir).iterdir():
        print("已经处理文件：{}".format(path))
        if not path.name.startswith('layer'):
            continue
        small_static_dict = torch.load(path, map_location="cpu")
        layer_i = int(path.name.split('-')[0].replace('layer_', ''))
        if layer_i == 0:
            model_static_dict["transformer.word_embeddings.weight"] = small_static_dict["word_embeddings.weight"]
        elif layer_i == 30:
            model_static_dict["lm_head.weight"] = small_static_dict["word_embeddings.weight"]
        elif layer_i == 29:
            for k, v in small_static_dict.items():
                model_static_dict["transformer." + k] = v
        else:
            for k, v in small_static_dict.items():
                model_static_dict["transformer." + k.replace("layer.", "layers.{}.".format(layer_i - 1))] = v

    torch.save(model_static_dict, join(save_model_dir, "pytorch_model.bin"))
    copy(join(ori_model_dir, "config.json"), join(save_model_dir, "config.json"))
    copy(join(ori_model_dir, "tokenizer_config.json"), join(save_model_dir, "tokenizer_config.json"))
    copy(join(ori_model_dir, "ice_text.model"), os.path.join(save_model_dir, "ice_text.model"))


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_model_dir', default='ChatGLM-6B/', type=str, help='')
    parser.add_argument('--pipeline_model_dir', default='output-glm-pp/global_step300/', type=str, help='')
    parser.add_argument('--save_model_dir', default='output-glm-pp/gs300/', type=str, help='')
    return parser.parse_args()


if __name__ == '__main__':
    ages = set_args()
    convert_model_to_hf(ages.ori_model_dir, ages.pipeline_model_dir, ages.save_model_dir)
