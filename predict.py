# -*- coding:utf-8 -*-
# @project: ChatGLM-Finetuning
# @filename: predict
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/12/6 20:41
"""
    文件说明：
            
"""
import argparse
import torch
from model import MODE


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--device", type=str, default="0", help="")
    parser.add_argument("--mode", type=str, default="glm3", help="")
    parser.add_argument("--model_path", type=str, default="output_dir/", help="")
    parser.add_argument("--max_length", type=int, default=2048, help="")
    parser.add_argument("--do_sample", type=bool, default=True, help="")
    parser.add_argument("--top_p", type=float, default=0.8, help="")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    return parser.parse_args()


def predict_one_sample(instruction, input, model, tokenizer, args):
    result, _ = model.chat(tokenizer, instruction + input, max_length=args.max_length, do_sample=args.do_sample,
                           top_p=args.top_p, temperature=args.temperature)
    return result


if __name__ == '__main__':
    args = parse_args()
    model = MODE[args.mode]["model"].from_pretrained(args.model_path, device_map="cuda:{}".format(args.device),
                                                     torch_dtype=torch.float16)
    tokenizer = MODE[args.mode]["tokenizer"].from_pretrained(args.model_path)
    instruction = "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\n分割。文本："
    input = "故障现象：发动机水温高，风扇始终是低速转动，高速档不工作，开空调尤其如此。"
    r = predict_one_sample(instruction, input, model, tokenizer, args)
    print(r)



