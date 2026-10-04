"""Calibration data for WikiText-2 and SQuAD."""
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def get_wikitext2(nsamples, seed, seqlen, model):
    print("get_wikitext2")
    # Lucifer Li: 使用`datasets`库加载数据集
    # Lucifer Li: `traindata`是一个`Dataset`对象, 可以像操作列表一样通过索引访问每个样本, 每个样本是一个字典, 结构为{'text': 'some text...'}
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    # Lucifer Li: 根据`model`指定分词器, model > `facebook/opt-125m`
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    # Lucifer Li: 极有可能是旧版的`datasets`库, 推测`traindata['text']`为字符串列表
    # Lucifer Li: `trainenc`一般包含`input_ids`和`attention_masks`
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    
    random.seed(seed)
    # Lucifer Li: 构建一个用于因果语言建模的数据集, 目标是让模型根据给定的上文预测下一个词
    trainloader = []
    # Lucifer Li: 循环nsamples次
    for _ in range(nsamples):
        # Lucifer Li: 每次从庞大的原始语料(trainenc.input_ids)中随机选取一个起始点i，然后截取从i到i+seqlen的连续token序列作为输入inp
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        # Lucifer Li: tar中除了最后一个位置之外的所有token都设置为`-100`, 在PyTorch的交叉熵损失函数中, `-100`是一个特殊值, 表示忽略该位置的损失计算
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    # Lucifer Li: 设置成功后, 模型需要学习的任务是: see token1, predict token2; see token1, token2, ..., tokenN-1, predict tokenN;
    return trainloader, None



def _format_squad_example(example, include_answer=True):
    """
    将 HuggingFace squad 样本转成 causal LM 校准用的纯文本。
    这里刻意不复用 tasks.py 的 SQuADDataset，避免 datautils.py 和 tasks.py 耦合。
    """
    title = example["title"]
    context = example["context"]
    question = example["question"].strip()

    answers = example["answers"]["text"]
    answer = answers[0] if len(answers) > 0 else ""

    text = f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer:"
    if include_answer:
        text += f" {answer}"
    return text



def get_squad(nsamples, seed, seqlen, model):
    print("get_squad")

    traindata = load_dataset("squad", split="train")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # 采用和 tasks.py 的 SQuADv2Template 相近的 prompt 形式：
    # Title / Context / Question / Answer
    train_text = "\n\n".join(
        _format_squad_example(example, include_answer=True)
        for example in traindata
    )
    trainenc = tokenizer(train_text, return_tensors="pt")

    if trainenc.input_ids.shape[1] <= seqlen:
        raise ValueError(
            f"SQuAD token length {trainenc.input_ids.shape[1]} is <= seqlen {seqlen}; "
            "cannot sample calibration windows."
        )

    random.seed(seed)
    trainloader = []

    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen

        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100

        trainloader.append((inp, tar))

    return trainloader, None



def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    if name == 'wikitext2':
        return get_wikitext2(nsamples, seed, seqlen, model)
    if name.lower() == 'squad':
        return get_squad(nsamples, seed, seqlen, model)
    raise ValueError(f'Unsupported calibration dataset: {name}')
