# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: predict_d2q
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/17 15:45
"""
    文件说明：
            
"""
import torch
import json
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
from peft import PeftModel
from tqdm import tqdm
import time
import os
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='data/d2q_1.json', type=str, help='')
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--method', default='freeze', type=str, help='')
    parser.add_argument('--ori_model_dir', default='/data/work/lcong/public_model_path/ChatGLM-6B/', type=str, help='')
    parser.add_argument('--model_dir', default="output_dir_freeze/global_step-7855/", type=str, help='')
    parser.add_argument('--max_len', type=int, default=768, help='')
    parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--prompt_text', type=str, default="你现在是一个问题生成模型，请根据下面文档生成一个问题，文档：",
                        help='')
    parser.add_argument('--top_p', type=float, default=0.95, help='')
    parser.add_argument('--do_sample', type=bool, default=True, help='')
    parser.add_argument('--num_return_sequences', type=int, default=4, help='')
    parser.add_argument('--save_path', type=str, default="d2q_result_data/d2q_freeze.json", help='')
    return parser.parse_args()


def main():
    args = set_args()
    if args.method == "lora":
        model = ChatGLMForConditionalGeneration.from_pretrained(args.ori_model_dir)
        tokenizer = ChatGLMTokenizer.from_pretrained(args.model_dir)
        model.eval()
        model = PeftModel.from_pretrained(model, args.model_dir, torch_dtype=torch.float32)
        model.half().to("cuda:{}".format(args.device))
    else:
        model = ChatGLMForConditionalGeneration.from_pretrained(args.model_dir)
        tokenizer = ChatGLMTokenizer.from_pretrained(args.model_dir)
        model.half().to("cuda:{}".format(args.device))
        model.eval()

    save_data = []
    max_tgt_len = args.max_len - args.max_src_len - 3
    s_time = time.time()
    with open(args.test_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(tqdm(fh, desc="iter")):
            with torch.no_grad():
                sample = json.loads(line.strip())
                src_tokens = tokenizer.tokenize(sample["text"])
                prompt_tokens = tokenizer.tokenize(args.prompt_text)

                if len(src_tokens) > args.max_src_len - len(prompt_tokens):
                    src_tokens = src_tokens[:args.max_src_len - len(prompt_tokens)]

                tokens = prompt_tokens + src_tokens + ["[gMASK]", "<sop>"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                # input_ids = tokenizer.encode("帮我写个快排算法")
                input_ids = torch.tensor([input_ids]).to("cuda:{}".format(args.device))
                generation_kwargs = {
                    "min_length": 5,
                    "max_new_tokens": max_tgt_len,
                    "top_p": args.top_p,
                    "do_sample": args.do_sample,
                    "num_return_sequences": args.num_return_sequences,
                }
                if args.method == "lora":
                    response = model.generate_one(input_ids, **generation_kwargs)
                else:
                    response = model.generate(input_ids, **generation_kwargs)
                res = []
                for i_r in range(generation_kwargs["num_return_sequences"]):
                    outputs = response.tolist()[i_r][input_ids.shape[1]:]
                    r = tokenizer.decode(outputs).replace("<eop>", "")
                    res.append(r)
                save_data.append({"text": sample["text"], "ori_answer": sample["answer"], "gen_answer": res})
    e_time = time.time()
    print("总耗时：{}s".format(e_time - s_time))
    fin = open(args.save_path, "w", encoding="utf-8")
    json.dump(save_data, fin, ensure_ascii=False, indent=4)
    fin.close()


if __name__ == '__main__':
    main()
