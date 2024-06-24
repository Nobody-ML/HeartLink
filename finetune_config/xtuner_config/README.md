# 基于 Xtuner 的微调指南

## 1. 环境准备

- 微调硬件：对于 7B 的模型需要 A100(40G)


- 使用 conda 先构建一个 Python-3.10 的虚拟环境

  ```bash
  conda create --name finetune_xtuner python=3.10 -y
  conda activate finetune_xtuner
  ```

- 通过 pip 安装 XTuner：

  ```shell
  pip install -U xtuner
  ```

  亦可集成 DeepSpeed 安装：

  ```shell
  pip install -U 'xtuner[deepspeed]'
  ```

- 从源码安装 XTuner：

  ```shell
  git clone https://github.com/InternLM/xtuner.git
  cd xtuner
  pip install -e '.[all]'
  ```


## 2. 微调配置
- 进入配置文件夹

```shell
cd xtuner_config
```
- 查看 XTuner 支持模型
```shell
xtuner list-cfg
```
- 拷贝目标模型的配置文件与修改配置文件参数
```shell
xtuner copy-cfg internlm2_chat_7b_qlora_oasst1_e3 .
mv internlm2_chat_7b_qlora_oasst1_e3.py internlm2_chat_7b_qlora.py
vim internlm2_chat_7b_qlora.py
```

## 3. 微调训练
- 运行微调训练
```shell
# 单卡
xtuner train /root/SoulStar/finetune_config/xtuner_config/internlm2_chat_7b_qlora.py --deepspeed deepspeed_zero2
```
- `--deepspeed` 表示使用 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 来优化训练过程。XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3 等。如果用户期望关闭此功能，请直接移除此参数。

- 更多示例，请查阅[文档](https://github.com/InternLM/xtuner/blob/main/docs/zh_cn/user_guides/finetune.md)。

## 4. 模型参数转换合并
- pth 格式参数转换为 huggingface 格式
```shell
# 创建用于存放Hugging Face格式参数的hf文件夹
mkdir /root/SoulStar/finetune_config/xtuner_config/work_dirs/hf

export MKL_SERVICE_FORCE_INTEL=1

# 配置文件存放的位置
export CONFIG_NAME_OR_PATH=/root/SoulStar/finetune_config/xtuner_config/internlm2_chat_7b_qlora.py

# 模型训练后得到的pth格式参数存放的位置
export PTH=/root/SoulStar/finetune_config/xtuner_config/work_dirs/internlm2_chat_7b_qlora/iter_2500.pth

# pth文件转换为Hugging Face格式后参数存放的位置
export SAVE_PATH=/root/SoulStar/finetune_config/xtuner_config/work_dirs/hf

# 执行参数转换
xtuner convert pth_to_hf $CONFIG_NAME_OR_PATH $PTH $SAVE_PATH
```

- huggingface 格式参数合并
```shell
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

# 原始模型参数存放的位置
export NAME_OR_PATH_TO_LLM=/root/model/Shanghai_AI_Laboratory/internlm2-chat-7b

# Hugging Face格式参数存放的位置
export NAME_OR_PATH_TO_ADAPTER=/root/SoulStar/finetune_config/xtuner_config/work_dirs/hf

# 最终Merge后的参数存放的位置
mkdir /root/model/internlm2-chat-7b-soulstar
export SAVE_PATH=/root/model/internlm2-chat-7b-soulstar

# 执行参数Merge
xtuner convert merge \
    $NAME_OR_PATH_TO_LLM \
    $NAME_OR_PATH_TO_ADAPTER \
    $SAVE_PATH \
    --max-shard-size 2GB
```
