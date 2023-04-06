
## ChatGLM微调
本项目主要针对ChatGLM模型进行不同方式的微调，并对比大模型在不同微调方法上的效果，主要针对信息抽取任务、生成任务、分类任务等。

为了模型适配其他方法，对官方ChatGLM模型文件进行了部分修改，将820-821行参数冻结代码删掉，再外部进行参数冻结。

上述实验结果均基于单卡训练。
## 微调方法
### Freeze方法
Freeze方法，即参数冻结，对原始模型部分参数进行冻结操作，仅训练部分参数，以达到在单卡或不进行TP或PP操作，就可以对大模型进行训练。

微调代码，见finetuning_freeze.py，核心部分如下：
```python3
for name, param in model.named_parameters():
    if not any(nd in name for nd in ["layers.27", "layers.26", "layers.25", "layers.24", "layers.23"]):
        param.requires_grad = False
```
针对模型不同层进行修改，可以自行修改。
训练代码均采用DeepSpeed进行训练，可设置参数包含train_path、model_dir、num_train_epochs、train_batch_size、gradient_accumulation_steps、output_dir、prompt_text等，
可根据自己的任务配置。
```
CUDA_VISIBLE_DEVICES=0 deepspeed finetuning_freeze.py --num_train_epochs 5 --train_batch_size 2
```
三元组抽取的推理代码，见predict_freeze.py，其他任务可以根据自己的评价标准进行推理预测。

### PT方法
PT方法，即P-Tuning方法，参考[ChatGLM官方代码](https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README.md) ，是一种针对于大模型的soft-prompt方法。

![](images/PT.png)
- P-Tuning仅对大模型的Embedding加入新的参数。[paper](https://arxiv.org/abs/2103.10385)
- P-Tuning-V2，将大模型的Embedding和每一层前都加上新的参数。[paper](https://arxiv.org/abs/2110.07602)
微调代码，见finetuning_pt.py，核心部分如下：
```python3
config = ChatGLMConfig.from_pretrained(args.model_dir)
config.pre_seq_len = args.pre_seq_len
config.prefix_projection = args.prefix_projection

model = ChatGLMForConditionalGeneration.from_pretrained(args.model_dir, config=config)

for name, param in model.named_parameters():
    if not any(nd in name for nd in ["prefix_encoder"]):
        param.requires_grad = False
```
当prefix_projection为True时，为P-Tuning-V2方法，在大模型的Embedding和每一层前都加上新的参数；为False时，为P-Tuning方法，仅在大模型的Embedding上新的参数。

训练代码均采用DeepSpeed进行训练，可设置参数包含train_path、model_dir、num_train_epochs、train_batch_size、gradient_accumulation_steps、output_dir、prompt_text、pre_seq_len、prompt_text等，
可根据自己的任务配置。
```
CUDA_VISIBLE_DEVICES=0 deepspeed finetuning_pt.py --num_train_epochs 5 --train_batch_size 2 --pre_seq_len 16
```
三元组抽取的推理代码，见predict_pt.py，其他任务可以根据自己的评价标准进行推理预测。

### Lora方法
Lora方法，即在大型语言模型上对指定参数增加额外的低秩矩阵，并在模型训练过程中，仅训练而外增加的参数。
当“秩值”远小于原始参数维度时，新增的低秩矩阵参数量很小，达到仅训练很小的参数，就能获取较好的结果。

![](images/Lora.png)
- 论文：[paper](https://arxiv.org/abs/2106.09685)
- 官方代码：[Github](https://github.com/microsoft/LoRA)
- HuggingFace封装的peft库：[Github](https://github.com/huggingface/peft)

微调代码，见finetuning_lora.py，核心部分如下：
```python3
model = ChatGLMForConditionalGeneration.from_pretrained(args.model_dir)
config = LoraConfig(r=args.lora_r,
                    lora_alpha=32,
                    target_modules=["query_key_value"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM",
                    inference_mode=False,
                    )

model = get_peft_model(model, config)
```
训练代码均采用DeepSpeed进行训练，可设置参数包含train_path、model_dir、num_train_epochs、train_batch_size、gradient_accumulation_steps、output_dir、prompt_text、lora_r等，
可根据自己的任务配置。
```
CUDA_VISIBLE_DEVICES=0 deepspeed finetuning_lora.py --num_train_epochs 5 --train_batch_size 2 --lora_r 8
```
三元组抽取的推理代码，见predict_lora.py，其他任务可以根据自己的评价标准进行推理预测。

注意：对于结果需要保持一致的任务，需要保存模型的adapter_config.json文件中，inference_mode参数修改成false，并将模型执行model.eval()操作。
主要原因是chatglm模型代码中，没有采用Conv1D函数。

### 运行环境
查看requirements.txt文件

## 实验结果
### 三元组抽取
- 为了防止大模型的数据泄露，采用一个领域比赛数据集-[汽车工业故障模式关系抽取](https://www.datafountain.cn/competitions/584)，随机抽取50条作为测试集
- 模型训练时，最大长度为768，Batch大小为2，训练轮数为5，fp16训练，采用DeepSpeed的Zero-1训练；
- PT为官方的P-Tuning V2训练方法，PT-Only-Embedding表示仅对Embedding进行soft-prompt，Freeze仅训练模型后五层参数，Lora采用低秩矩阵方法训练，秩为8；
- 由于之间训练PT在48G-A40显卡上会出现OOM，因此下面PT实验对模型开启了gradient_checkpointing_enable()，使得模型显存占用变小，但训练时长增加。
- 训练示例：
```
prompt_text：你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：
输入：故障现象：发动机水温高，风扇始终是低速转动，高速档不工作，开空调尤其如此。
输出：发动机_部件故障_水温高\n风扇_部件故障_低速转动
```


| 微调方法 |  PT-Only-Embedding |  PT | Freeze |  Lora | 
| ------- | ------ | ------  | ------ | ------ |
| 显卡占用 | 37G | 30G | 24G | 39G |
| 总参数 | 62.59B | 72.11B | 62.55B | 62.59B |
| 可训练参数占比 | 0.0586% | 13.26% | 16.10% | 0.0586% |
| 训练耗时 | 53min | 135min | 112min | 65min |
| 测试结果F1 | 0.0 | 0.6283 | 0.5675 | 0.5359 |
| 测试耗时 | 191s | 198s | 180s | 278s |

结构分析：
- 效果为PT>Freeze>Lora>PT-Only-Embedding
- PT-Only-Embedding效果很不理想，发现在训练时，最后的loss仅能收敛到2.几，而其他机制可以收敛到0.几。分析原因为，输出内容形式与原有语言模型任务相差很大，仅增加额外Embedding参数，不足以改变复杂的下游任务。
- PT方法占用显存更大，因为也增加了很多而外参数。
- 测试耗时，由于其他方法均增加了额外参数，因此推理耗时会比Freeze方法要高。
- 上面测试仅代表个人测试结果，并且由于生成模型生成长度对推理耗时影响很大，因此可以其他数据会有不一样的结果。
- 模型在指定任务上微调之后，并没有丧失原有能力，例如生成“帮我写个快排算法”，依然快排代码。

### 文本生成
待补充

### 文本分类
待补充