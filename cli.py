import os

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from glm2.modeling_chatglm import ChatGLMForConditionalGeneration
from glm2.tokenization_chatglm import ChatGLMTokenizer
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')

    parser.add_argument('--max_len', type=int, default=1560, help='')
    parser.add_argument('--max_src_len', type=int, default=1024, help='')
    parser.add_argument('--top_p', type=float, default=0.7, help='')
    parser.add_argument('--do_sample', type=bool, default=False, help='')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='')
    parser.add_argument('--model_dir',
                        default="/home/workspace/from_pretrained/chatglm2-6b", type=str,
                        help='')
    parser.add_argument('--ptuning_checkpoint',
                        default="/home/workspace/code_server/fastllm/ChatGLM-Finetuning/output-glm2-0906/epoch-1-step-45",
                        type=str,
                        help='')
    parser.add_argument('--pre_seq_len', type=int, default=16, help='')
    parser.add_argument('--prefix_projection', type=bool, default=True, help='')

    return parser.parse_args()


def predict_one_sample(model, tokenizer, args, text):
    max_tgt_len = args.max_len - args.max_src_len - 3
    with torch.no_grad():
        text="[Round {}]\n\n问：{}\n\n答：".format(1, text)
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

    tokenizer = ChatGLMTokenizer.from_pretrained(args.ptuning_checkpoint)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    config.pre_seq_len = args.pre_seq_len
    config.prefix_projection = args.prefix_projection

    model = AutoModel.from_pretrained(args.model_dir, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(args.ptuning_checkpoint, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("module.transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("module.transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model.transformer.prefix_encoder.float()

    model.half().to("cuda:{}".format(args.device))
    model.eval()


    print('开始进行问答，输入CTRL + C，则退出')
    while True:
        print('问：')
        text = input()
        pre_res = predict_one_sample(model, tokenizer, args, text)
        print("答：{}".format(pre_res))


if __name__ == '__main__':
    main()
