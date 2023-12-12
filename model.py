# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: model
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/8/6 16:13
"""
    文件说明：
            
"""
from glm3.modeling_chatglm import ChatGLMForConditionalGeneration as ChatGLM3ForConditionalGeneration
from glm3.tokenization_chatglm import ChatGLMTokenizer as ChatGLM3Tokenizer
from glm3.configuration_chatglm import ChatGLMConfig as ChatGLM3Config
from glm2.modeling_chatglm import ChatGLMForConditionalGeneration as ChatGLM2ForConditionalGeneration
from glm2.tokenization_chatglm import ChatGLMTokenizer as ChatGLM2Tokenizer
from glm2.configuration_chatglm import ChatGLMConfig as ChatGLM2Config
from glm1.modeling_chatglm import ChatGLMForConditionalGeneration
from glm1.tokenization_chatglm import ChatGLMTokenizer
from glm1.configuration_chatglm import ChatGLMConfig
from utils import GLMPromptDataSet, GLM2PromptDataSet, GLM3PromptDataSet

MODE = {"glm": {"model": ChatGLMForConditionalGeneration, "tokenizer": ChatGLMTokenizer, "config": ChatGLMConfig,
                "dataset": GLMPromptDataSet},
        "glm2": {"model": ChatGLM2ForConditionalGeneration, "tokenizer": ChatGLM2Tokenizer, "config": ChatGLM2Config,
                 "dataset": GLM2PromptDataSet},
        "glm3": {"model": ChatGLM3ForConditionalGeneration, "tokenizer": ChatGLM3Tokenizer, "config": ChatGLM3Config,
                 "dataset": GLM3PromptDataSet}
        }
