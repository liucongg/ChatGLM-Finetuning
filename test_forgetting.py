# -*- coding:utf-8 -*-
# @project: ChatGLM-Finetuning
# @filename: test_forgetting
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/7 15:00
"""
    文件说明：
            
"""
import torch
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--model_dir', default="/data/work/lcong/ChatGPT/LLMFTProj/output_dir_freeze/global_step-2160/",
                        type=str, help='')
    parser.add_argument('--max_len', type=int, default=2048, help='')
    parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--top_p', type=float, default=0.7, help='')
    parser.add_argument('--do_sample', type=bool, default=True, help='')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='')
    return parser.parse_args()


def predict_one_sample(model, tokenizer, args, text):
    max_tgt_len = args.max_len - args.max_src_len - 3
    with torch.no_grad():
        input_ids = tokenizer.encode(text, max_length=args.max_src_len, truncation=True)
        input_ids = torch.tensor([input_ids]).to("cuda:{}".format(args.device))
        generation_kwargs = {
            "min_length": 5,
            "max_new_tokens": max_tgt_len,
            "top_p": args.top_p,
            "temperature": 0.95,
            "do_sample": args.do_sample,
            "num_return_sequences": args.num_return_sequences,
        }
        response = model.generate(input_ids, **generation_kwargs)

        res = []
        for i_r in range(generation_kwargs["num_return_sequences"]):
            outputs = response.tolist()[i_r][input_ids.shape[1]:]
            r = tokenizer.decode(outputs).replace("<eop>", "")
            res.append(r)
    return res[0]


def main():
    args = set_args()
    model = ChatGLMForConditionalGeneration.from_pretrained(args.model_dir)
    model.half().to("cuda:{}".format(args.device))
    model.eval()
    tokenizer = ChatGLMTokenizer.from_pretrained(args.model_dir)

    print('开始进行问答，输入CTRL + C，则退出')
    while True:
        text = input("问：")
        pre_res = predict_one_sample(model, tokenizer, args, text)
        print("答：{}".format(pre_res))


if __name__ == '__main__':
    main()
