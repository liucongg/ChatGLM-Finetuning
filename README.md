
## ChatGLM微调
本项目主要针对ChatGLM和ChatGLM2模型进行不同方式的微调（Freeze方法、Lora方法、P-Tuning方法、全量参数等），并对比大模型在不同微调方法上的效果，主要针对信息抽取任务、生成任务、分类任务等。

本项目支持单卡训练&多卡训练，由于采用单指令集方式微调，模型微调之后**并没有出现严重的灾难性遗忘**。

由于官方代码和模型一直在更新，目前代码和模型使用的是最新版本（20230806）。

PS：没有用Trainer（虽然Trainer代码简单，但不易修改，大模型时代算法工程师本就成为了数据工程师，因此更需了解训练流程）

## 更新简介
- update-2023.08.06 代码和模型已经更新到最新，支持单卡&多卡训练，支持ChatGLM2模型训练、支持全量参数训练，所有代码进行了结构增加可读性。
- update-2023.06.12 [**增加流水线并行训练方法**](https://zhuanlan.zhihu.com/p/636488690)，请看[v0.1 Tag](https://github.com/liucongg/ChatGLM-Finetuning/tree/v0.1)
- update-2023.04.18 **增加文本生成任务评测**，请看[v0.1 Tag](https://github.com/liucongg/ChatGLM-Finetuning/tree/v0.1)
- update-2023.04.05 **增加信息抽取任务评测**，请看[v0.1 Tag](https://github.com/liucongg/ChatGLM-Finetuning/tree/v0.1)

## 微调方法
模型微调时，如果遇到显存不够的情况，可以开启gradient_checkpointing、zero3、offload等参数来节省显存。

下面model_name_or_path参数为模型路径，请根据可根据自己实际模型保存地址进行修改。
### Freeze方法
Freeze方法，即参数冻结，对原始模型部分参数进行冻结操作，仅训练部分参数，以达到在单卡或多卡，不进行TP或PP操作就可以对大模型进行训练。

微调代码，见train.py，核心部分如下：
```python3
freeze_module_name = args.freeze_module_name.split(",")
for name, param in model.named_parameters():
	if not any(nd in name for nd in freeze_module_name):
		param.requires_grad = False
```
针对模型不同层进行修改，可以自行修改freeze_module_name参数配置，例如"layers.27.,layers.26.,layers.25.,layers.24."。
训练代码均采用DeepSpeed进行训练，可设置参数包含train_path、model_name_or_path、mode、train_type、freeze_module_name、ds_file、num_train_epochs、per_device_train_batch_size、gradient_accumulation_steps、output_dir等， 可根据自己的任务配置。

ChatGLM单卡训练
```
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 520 train.py \
                --train_path data/spo_0.json \
                --model_name_or_path ChatGLM-6B/ \
                --per_device_train_batch_size 1 \
                --max_len 1560 \
                --max_src_len 1024 \
                --learning_rate 1e-4 \
                --weight_decay 0.1 \
                --num_train_epochs 2 \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --mode glm \
                --train_type freeze \
                --freeze_module_name "layers.27.,layers.26.,layers.25.,layers.24." \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --output_dir ./output-glm
```
ChatGLM四卡训练，通过CUDA_VISIBLE_DEVICES控制具体哪几块卡进行训练，如果不加该参数，表示使用运行机器上所有卡进行训练
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 520 train.py \
                --train_path data/spo_0.json \
                --model_name_or_path ChatGLM-6B/ \
                --per_device_train_batch_size 1 \
                --max_len 1560 \
                --max_src_len 1024 \
                --learning_rate 1e-4 \
                --weight_decay 0.1 \
                --num_train_epochs 2 \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --mode glm \
                --train_type freeze \
                --freeze_module_name "layers.27.,layers.26.,layers.25.,layers.24." \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --output_dir ./output-glm
```
ChatGLM2单卡训练
```
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 520 train.py \
                --train_path data/spo_0.json \
                --model_name_or_path ChatGLM2-6B/ \
                --per_device_train_batch_size 1 \
                --max_len 1560 \
                --max_src_len 1024 \
                --learning_rate 1e-4 \
                --weight_decay 0.1 \
                --num_train_epochs 2 \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --mode glm2 \
                --train_type freeze \
                --freeze_module_name "layers.27.,layers.26.,layers.25.,layers.24." \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --output_dir ./output-glm2
```
ChatGLM2四卡训练，通过CUDA_VISIBLE_DEVICES控制具体哪几块卡进行训练，如果不加该参数，表示使用运行机器上所有卡进行训练
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 520 train.py \
                --train_path data/spo_0.json \
                --model_name_or_path ChatGLM2-6B/ \
                --per_device_train_batch_size 1 \
                --max_len 1560 \
                --max_src_len 1024 \
                --learning_rate 1e-4 \
                --weight_decay 0.1 \
                --num_train_epochs 2 \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --mode glm2 \
                --train_type freeze \
                --freeze_module_name "layers.27.,layers.26.,layers.25.,layers.24." \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --output_dir ./output-glm2
```
PS：ChatGLM微调时所用显存要比ChatGLM2多，详细显存占比如下：

| Model |  DeepSpeed-Stage |  Offload | Gradient Checkpointing |  Batch Size | Max Length | GPU-A40 Number | 所耗显存 |
| ------- | ------ | ------  | ------ | ------ | ------  | ------ | ------ |
| ChaGLM | zero2 | No | Yes | 1 | 1560  | 1 | 36G |
| ChaGLM | zero2 | No | No | 1 | 1560  | 1 | 38G |
| ChaGLM | zero2 | No | Yes | 1 | 1560  | 4 | 24G |
| ChaGLM | zero2 | No | No | 1 | 1560  | 4 | 29G |
| ChaGLM2 | zero2 | No | Yes | 1 | 1560  | 1 | 35G |
| ChaGLM2 | zero2 | No | No | 1 | 1560  | 1 | 36G |
| ChaGLM2 | zero2 | No | Yes | 1 | 1560  | 4 | 22G |
| ChaGLM2 | zero2 | No | No | 1 | 1560  | 4 | 27G |


### PT方法
PT方法，即P-Tuning方法，参考[ChatGLM官方代码](https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README.md) ，是一种针对于大模型的soft-prompt方法。

![](images/PT.png)
- P-Tuning仅对大模型的Embedding加入新的参数。[paper](https://arxiv.org/abs/2103.10385)
- P-Tuning-V2，将大模型的Embedding和每一层前都加上新的参数。[paper](https://arxiv.org/abs/2110.07602)

微调代码，见train.py，核心部分如下：
```python3
config = MODE[args.mode]["config"].from_pretrained(args.model_name_or_path)
config.pre_seq_len = args.pre_seq_len
config.prefix_projection = args.prefix_projection
model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path, config=config)
for name, param in model.named_parameters():
	if not any(nd in name for nd in ["prefix_encoder"]):
		param.requires_grad = False
```
当prefix_projection为True时，为P-Tuning-V2方法，在大模型的Embedding和每一层前都加上新的参数；为False时，为P-Tuning方法，仅在大模型的Embedding上新的参数。

训练代码均采用DeepSpeed进行训练，可设置参数包含train_path、model_name_or_path、mode、train_type、pre_seq_len、prefix_projection、ds_file、num_train_epochs、per_device_train_batch_size、gradient_accumulation_steps、output_dir等， 可根据自己的任务配置。

ChatGLM单卡训练
```
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 520 train.py \
                --train_path data/spo_0.json \
                --model_name_or_path ChatGLM-6B \
                --per_device_train_batch_size 1 \
                --max_len 768 \
                --max_src_len 512 \
                --learning_rate 1e-4 \
                --weight_decay 0.1 \
                --num_train_epochs 2 \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --mode glm \
                --train_type ptuning \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --pre_seq_len 16 \
                --prefix_projection True \
                --output_dir ./output-glm
```
ChatGLM四卡训练，通过CUDA_VISIBLE_DEVICES控制具体哪几块卡进行训练，如果不加该参数，表示使用运行机器上所有卡进行训练
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 520 train.py \
                --train_path data/spo_0.json \
                --model_name_or_path ChatGLM-6B \
                --per_device_train_batch_size 1 \
                --max_len 1560 \
                --max_src_len 1024 \
                --learning_rate 1e-4 \
                --weight_decay 0.1 \
                --num_train_epochs 2 \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --mode glm \
                --train_type ptuning \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --pre_seq_len 16 \
                --prefix_projection True \
                --output_dir ./output-glm
```
ChatGLM2单卡训练
```
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 520 train.py \
                --train_path data/spo_0.json \
                --model_name_or_path ChatGLM2-6B \
                --per_device_train_batch_size 1 \
                --max_len 1560 \
                --max_src_len 1024 \
                --learning_rate 1e-4 \
                --weight_decay 0.1 \
                --num_train_epochs 2 \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --mode glm2 \
                --train_type ptuning \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --pre_seq_len 16 \
                --prefix_projection True \
                --output_dir ./output-glm2
```
ChatGLM2四卡训练，通过CUDA_VISIBLE_DEVICES控制具体哪几块卡进行训练，如果不加该参数，表示使用运行机器上所有卡进行训练
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 520 train.py \
                --train_path data/spo_0.json \
                --model_name_or_path ChatGLM2-6B \
                --per_device_train_batch_size 1 \
                --max_len 1560 \
                --max_src_len 1024 \
                --learning_rate 1e-4 \
                --weight_decay 0.1 \
                --num_train_epochs 2 \
                --gradient_accumulation_steps 4 \
                --warmup_ratio 0.1 \
                --mode glm2 \
                --train_type ptuning \
                --seed 1234 \
                --ds_file ds_zero2_no_offload.json \
                --gradient_checkpointing \
                --show_loss_step 10 \
                --pre_seq_len 16 \
                --prefix_projection True \
                --output_dir ./output-glm2
```
PS：ChatGLM微调时所用显存要比ChatGLM2多，详细显存占比如下：

| Model |  DeepSpeed-Stage |  Offload | Gradient Checkpointing |  Batch Size | Max Length | GPU-A40 Number | 所耗显存 |
| ------- | ------ | ------  | ------ | ------ | ------  | ------ | ------ |
| ChaGLM | zero2 | No | Yes | 1 | 768  | 1 | 43G |
| ChaGLM | zero2 | No | No | 1 | 300  | 1 | 44G |
| ChaGLM | zero2 | No | Yes | 1 | 1560  | 4 | 37G |
| ChaGLM | zero2 | No | No | 1 | 1360  | 4 | 44G |
| ChaGLM2 | zero2 | No | Yes | 1 | 1560  | 1 | 20G |
| ChaGLM2 | zero2 | No | No | 1 | 1560  | 1 | 40G |
| ChaGLM2 | zero2 | No | Yes | 1 | 1560  | 4 | 19G |
| ChaGLM2 | zero2 | No | No | 1 | 1560  | 4 | 39G |


### Lora方法
Lora方法，即在大型语言模型上对指定参数（权重矩阵）并行增加额外的低秩矩阵，并在模型训练过程中，仅训练额外增加的并行低秩矩阵的参数。
当“秩值”远小于原始参数维度时，新增的低秩矩阵参数量也就很小。在下游任务tuning时，仅须训练很小的参数，但能获取较好的表现结果。

![](images/Lora.png)
- 论文：[paper](https://arxiv.org/abs/2106.09685)
- 官方代码：[Github](https://github.com/microsoft/LoRA)
- HuggingFace封装的peft库：[Github](https://github.com/huggingface/peft)

微调代码，见train.py，核心部分如下：
```python3
model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path)
lora_module_name = args.lora_module_name.split(",")
config = LoraConfig(r=args.lora_dim,
					lora_alpha=args.lora_alpha,
					target_modules=lora_module_name,
					lora_dropout=args.lora_dropout,
					bias="none",
					task_type="CAUSAL_LM",
					inference_mode=False,
					)
model = get_peft_model(model, config)
model.config.torch_dtype = torch.float32
```
训练代码均采用DeepSpeed进行训练，可设置参数包含train_path、model_name_or_path、mode、train_type、lora_dim、lora_alpha、lora_dropout、lora_module_name、ds_file、num_train_epochs、per_device_train_batch_size、gradient_accumulation_steps、output_dir等， 可根据自己的任务配置。

ChatGLM单卡训练
```
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 520 train.py \
              --train_path data/spo_0.json \
              --model_name_or_path ChatGLM-6B \
              --per_device_train_batch_size 1 \
              --max_len 1560 \
              --max_src_len 1024 \
              --learning_rate 1e-4 \
              --weight_decay 0.1 \
              --num_train_epochs 2 \
              --gradient_accumulation_steps 4 \
              --warmup_ratio 0.1 \
              --mode glm \
              --train_type lora \
              --lora_dim 16 \
              --lora_alpha 64 \
              --lora_dropout 0.1 \
              --lora_module_name "query_key_value" \
              --seed 1234 \
              --ds_file ds_zero2_no_offload.json \
              --gradient_checkpointing \
              --show_loss_step 10 \
              --output_dir ./output-glm
```
ChatGLM四卡训练，通过CUDA_VISIBLE_DEVICES控制具体哪几块卡进行训练，如果不加该参数，表示使用运行机器上所有卡进行训练
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 520 train.py \
              --train_path data/spo_0.json \
              --model_name_or_path ChatGLM-6B \
              --per_device_train_batch_size 1 \
              --max_len 1560 \
              --max_src_len 1024 \
              --learning_rate 1e-4 \
              --weight_decay 0.1 \
              --num_train_epochs 2 \
              --gradient_accumulation_steps 4 \
              --warmup_ratio 0.1 \
              --mode glm \
              --train_type lora \
              --lora_dim 16 \
              --lora_alpha 64 \
              --lora_dropout 0.1 \
              --lora_module_name "query_key_value" \
              --seed 1234 \
              --ds_file ds_zero2_no_offload.json \
              --gradient_checkpointing \
              --show_loss_step 10 \
              --output_dir ./output-glm
```
ChatGLM2单卡训练
```
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 520 train.py \
              --train_path data/spo_0.json \
              --model_name_or_path ChatGLM2-6B \
              --per_device_train_batch_size 1 \
              --max_len 1560 \
              --max_src_len 1024 \
              --learning_rate 1e-4 \
              --weight_decay 0.1 \
              --num_train_epochs 2 \
              --gradient_accumulation_steps 4 \
              --warmup_ratio 0.1 \
              --mode glm2 \
              --train_type lora \
              --lora_dim 16 \
              --lora_alpha 64 \
              --lora_dropout 0.1 \
              --lora_module_name "query_key_value,dense_h_to_4h,dense_4h_to_h,dense" \
              --seed 1234 \
              --ds_file ds_zero2_no_offload.json \
              --gradient_checkpointing \
              --show_loss_step 10 \
              --output_dir ./output-glm2
```
ChatGLM2四卡训练，通过CUDA_VISIBLE_DEVICES控制具体哪几块卡进行训练，如果不加该参数，表示使用运行机器上所有卡进行训练
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 520 train.py \
              --train_path data/spo_0.json \
              --model_name_or_path ChatGLM2-6B \
              --per_device_train_batch_size 1 \
              --max_len 1560 \
              --max_src_len 1024 \
              --learning_rate 1e-4 \
              --weight_decay 0.1 \
              --num_train_epochs 2 \
              --gradient_accumulation_steps 4 \
              --warmup_ratio 0.1 \
              --mode glm2 \
              --train_type lora \
              --lora_dim 16 \
              --lora_alpha 64 \
              --lora_dropout 0.1 \
              --lora_module_name "query_key_value,dense_h_to_4h,dense_4h_to_h,dense" \
              --seed 1234 \
              --ds_file ds_zero2_no_offload.json \
              --gradient_checkpointing \
              --show_loss_step 10 \
              --output_dir ./output-glm2
```
PS：ChatGLM微调时所用显存要比ChatGLM2多，详细显存占比如下：

| Model |  DeepSpeed-Stage |  Offload | Gradient Checkpointing |  Batch Size | Max Length | GPU-A40 Number | 所耗显存 |
| ------- | ------ | ------  | ------ | ------ | ------  | ------ | ------ |
| ChaGLM | zero2 | No | Yes | 1 | 1560  | 1 | 20G |
| ChaGLM | zero2 | No | No | 1 | 1560  | 1 | 45G |
| ChaGLM | zero2 | No | Yes | 1 | 1560  | 4 | 20G |
| ChaGLM | zero2 | No | No | 1 | 1560  | 4 | 45G |
| ChaGLM2 | zero2 | No | Yes | 1 | 1560  | 1 | 20G |
| ChaGLM2 | zero2 | No | No | 1 | 1560  | 1 | 43G |
| ChaGLM2 | zero2 | No | Yes | 1 | 1560  | 4 | 19G |
| ChaGLM2 | zero2 | No | No | 1 | 1560  | 4 | 42G |

注意：Lora方法在模型保存时仅保存了Lora训练参数，因此在模型预测时需要将模型参数进行合并，具体参考merge_lora.py。

### 全参方法
全参方法，对大模型进行全量参数训练，主要借助DeepSpeed-Zero3方法，对模型参数进行多卡分割，并借助Offload方法，将优化器参数卸载到CPU上以解决显卡不足问题。

微调代码，见train.py，核心部分如下：
```python3
model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path)
```
训练代码均采用DeepSpeed进行训练，可设置参数包含train_path、model_name_or_path、mode、train_type、ds_file、num_train_epochs、per_device_train_batch_size、gradient_accumulation_steps、output_dir等， 可根据自己的任务配置。

ChatGLM四卡训练，通过CUDA_VISIBLE_DEVICES控制具体哪几块卡进行训练，如果不加该参数，表示使用运行机器上所有卡进行训练
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 520 train.py \
              --train_path data/spo_0.json \
              --model_name_or_path ChatGLM-6B \
              --per_device_train_batch_size 1 \
              --max_len 1560 \
              --max_src_len 1024 \
              --learning_rate 1e-4 \
              --weight_decay 0.1 \
              --num_train_epochs 2 \
              --gradient_accumulation_steps 4 \
              --warmup_ratio 0.1 \
              --mode glm \
              --train_type all \
              --seed 1234 \
              --ds_file ds_zero3_offload.json \
              --gradient_checkpointing \
              --show_loss_step 10 \
              --output_dir ./output-glm
```
ChatGLM2四卡训练，通过CUDA_VISIBLE_DEVICES控制具体哪几块卡进行训练，如果不加该参数，表示使用运行机器上所有卡进行训练
```
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 520 train.py \
              --train_path data/spo_0.json \
              --model_name_or_path ChatGLM2-6B \
              --per_device_train_batch_size 1 \
              --max_len 1560 \
              --max_src_len 1024 \
              --learning_rate 1e-4 \
              --weight_decay 0.1 \
              --num_train_epochs 2 \
              --gradient_accumulation_steps 4 \
              --warmup_ratio 0.1 \
              --mode glm2 \
              --train_type all \
              --seed 1234 \
              --ds_file ds_zero3_no_offload.json \
              --gradient_checkpointing \
              --show_loss_step 10 \
              --output_dir ./output-glm2
```
PS：ChatGLM微调时所用显存要比ChatGLM2多，详细显存占比如下：

| Model |  DeepSpeed-Stage |  Offload | Gradient Checkpointing |  Batch Size | Max Length | GPU-A40 Number | 所耗显存 |
| ------- | ------ | ------  | ------ | ------ | ------  | ------ | ------ |
| ChaGLM | zero3 | Yes | Yes | 1 | 1560  | 4 | 33G |
| ChaGLM2 | zero3 | No | Yes | 1 | 1560  | 4 | 44G |
| ChaGLM2 | zero3 | Yes | Yes | 1 | 1560  | 4 | 26G |

后面补充DeepSpeed的Zero-Stage的相关内容说明。

### 运行环境
查看requirements.txt文件

## 实验结果
### 三元组抽取
- 为了防止大模型的数据泄露，采用一个领域比赛数据集-[汽车工业故障模式关系抽取](https://www.datafountain.cn/competitions/584)，随机抽取50条作为测试集
- 训练示例：
```
{
    "instruction": "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：",
    "input": "故障现象：发动机水温高，风扇始终是低速转动，高速档不工作，开空调尤其如此。",
    "output": "发动机_部件故障_水温高\n风扇_部件故障_低速转动"
}
```


| 微调方法 |  PT-Only-Embedding |  PT | Freeze |  Lora | 
| ------- | ------ | ------  | ------ | ------ |
| 测试结果F1 | 0.0 | 0.6283 | 0.5675 | 0.5359 |

结构分析：
- 效果为PT>Freeze>Lora>PT-Only-Embedding
- PT-Only-Embedding效果很不理想，发现在训练时，最后的loss仅能收敛到2.几，而其他机制可以收敛到0.几。分析原因为，输出内容形式与原有语言模型任务相差很大，仅增加额外Embedding参数，不足以改变复杂的下游任务。
- 上面测试仅代表个人测试结果，并且由于生成模型生成长度对推理耗时影响很大，因此可以其他数据会有不一样的结果。
- 模型在指定任务上微调之后，并没有丧失原有能力，例如生成“帮我写个快排算法”，依然可以生成-快排代码。
- 由于大模型微调都采用大量instruction进行模型训练，仅采用单一的指令进行微调时，对原来其他的指令影响不大，因此并没导致原来模型的能力丧失。

很多同学在微调后出现了灾难性遗忘现象，但本项目的训练代码并没有出现，对“翻译任务”、“代码任务”、“问答任务”进行测试，具体测试效果如下：
<details><summary><b>翻译任务</b></summary>

![](images/ft_fanyi.png)

</details>
<details><summary><b>代码任务</b></summary>

![](images/ft_code.png)

</details>
<details><summary><b>问答任务</b></summary>

![](images/ft_qa.png)

</details>

### 文本生成
- 为了防止大模型的数据泄露，采用一个“万创杯”中医药天池大数据竞赛-[中医文献问题生成挑战](https://tianchi.aliyun.com/competition/entrance/531826/introduction)，随机抽取20条作为测试集
- PT为官方的P-Tuning V2训练方法，PT-Only-Embedding表示仅对Embedding进行soft-prompt，Freeze仅训练模型后五层参数，Lora采用低秩矩阵方法训练，秩为8；
- 训练示例：
```
{
    "instruction": "你现在是一个问题生成模型，请根据下面文档生成一个问题，文档：",
    "input": "清热解毒口服液由生石膏、知母、紫花地丁、金银花、麦门冬、黄芩、玄参、连翘、龙胆草、生地黄、栀子、板蓝根组成。具有疏风解表、清热解毒利咽、生津止渴的功效，适用于治疗外感时邪、内有蕴热所致的身热汗出、头痛身痛、心烦口渴、微恶寒或反恶热、舌红、苔黄、脉数等症。现代临床主要用于治疗流行性感冒、流行性脑脊髓膜炎、肺炎等各种发热性疾病。口服液：每支10毫升，每次10~20毫升，每日3次。〔注意事项〕阳虚便澹者不宜使用。",
    "output": "清热解毒口服的功效有哪些？"
}
```

由于生成模型的内容不能想信息抽取任务一样评价，用现有的BLUE或者Rouge来评价也是不合适，因此制定了评分规则。 通过多样性和准确性两个角度判断D2Q模型好坏，每个样本总计5分，共20个样本。
- 多样性：
	- 问题是否高度相似，每重复一个问题扣0.25分；
	- 问题对应答案是否相同，每有一个重复答案或找不到答案，扣0.25分；
- 准确性：
	- 问题能否从文档中找到答案，每有一个找不到答案，扣0.25分；
	- 问题内容是否流畅，每有一个问题不流畅，扣0.25分；
	- 问题内容是否有害，每有一个有害，扣0.25分；

| 微调方法 |  原始模型 | PT-Only-Embedding |  PT | Freeze |  Lora | 
| ------- | ------ | ------ | ------  | ------ | ------ |
| 分数 | 51.75 | 73.75 | 87.75 | 79.25 | 86.75 |


## 流水线并行训练
代码说明见：[大模型流水线并行（Pipeline）实战](https://zhuanlan.zhihu.com/p/636488690)

请看[v0.1 Tag](https://github.com/liucongg/ChatGLM-Finetuning/tree/v0.1)

## Star History
![Star History Chart](https://api.star-history.com/svg?repos=liucongg/ChatGLM-Finetuning&type=Date)