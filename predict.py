import torch
import json

from transformers import AutoConfig, AutoTokenizer, AutoModel

from glm2.modeling_chatglm import ChatGLMForConditionalGeneration
from glm2.tokenization_chatglm import ChatGLMTokenizer
from tqdm import tqdm
import time
import os
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='data/spo_1.json', type=str, help='')
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--model_dir',
                        default="/home/workspace/from_pretrained/chatglm2-6b", type=str,
                        help='')
    parser.add_argument('--ptuning_checkpoint',
                        default="/home/workspace/code_server/fastllm/ChatGLM-Finetuning/output-glm2-0906/epoch-1-step-45",
                        type=str,
                        help='')
    parser.add_argument('--max_len', type=int, default=1560, help='')
    parser.add_argument('--max_src_len', type=int, default=1024, help='')
    parser.add_argument('--pre_seq_len', type=int, default=16, help='')
    parser.add_argument('--prefix_projection', type=bool, default=True, help='')
    parser.add_argument('--prompt_text', type=str,
                        default=r"你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\n分割。文本：",
                        help='')
    return parser.parse_args()


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

    save_data = []
    f1 = 0.0
    max_tgt_len = args.max_len - args.max_src_len - 3
    s_time = time.time()
    with open(args.test_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(tqdm(fh, desc="iter")):
            with torch.no_grad():
                sample = json.loads(line.strip())
                src_tokens = tokenizer.tokenize(
                    "[Round {}]\n\n问：{}\n\n答：".format(1, sample["instruction"] + sample["input"]))

                if len(src_tokens) > args.max_src_len:
                    # 当输入内容超长时，随向后截断，但保留“\n\n答：”内容
                    src_tokens = src_tokens[:args.max_src_len - 4] + src_tokens[-4:]
                # ChatGLM2需要增加[gMASK]、sop两个标记
                input_ids = [tokenizer.get_command("[gMASK]"),
                             tokenizer.get_command("sop")] + tokenizer.convert_tokens_to_ids(src_tokens)
                input_ids = torch.tensor([input_ids]).to("cuda:{}".format(args.device))
                generation_kwargs = {
                    "min_length": 5,
                    "max_new_tokens": max_tgt_len,
                    "top_p": 0.7,
                    "temperature": 0.95,
                    "do_sample": False,
                    "num_return_sequences": 1,
                }
                response = model.generate(input_ids, **generation_kwargs)
                res = []
                for i_r in range(generation_kwargs["num_return_sequences"]):
                    outputs = response.tolist()[i_r][input_ids.shape[1]:]
                    r = tokenizer.decode(outputs).replace("<eop>", "")
                    res.append(r)

                pre_res = [rr for rr in res[0].split("\n") if len(rr.split("_")) == 3]
                real_res = sample["output"].split("\n")
                same_res = set(pre_res) & set(real_res)
                if len(set(pre_res)) == 0:
                    p = 0.0
                else:
                    p = len(same_res) / len(set(pre_res))
                r = len(same_res) / len(set(real_res))
                if (p + r) != 0.0:
                    f = 2 * p * r / (p + r)
                else:
                    f = 0.0
                f1 += f
                print(f)
                save_data.append(
                    {"text": sample["input"], "ori_answer": sample["output"], "gen_answer": res[0], "f1": f})

    e_time = time.time()
    print("总耗时：{}s".format(e_time - s_time))
    print(f1 / 50)

    fin = open("data/ft_pt_answer.json", "w", encoding="utf-8")
    json.dump(save_data, fin, ensure_ascii=False, indent=4)
    fin.close()


if __name__ == '__main__':
    main()
