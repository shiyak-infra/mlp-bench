import os
import time

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

FLOPS = 423767310336

epoch_count = 1
batch_size = 32
n_embd = 2048
n_layer = 32
n_head = 16
n_positions = 1024


def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    local_rank = setup_distributed()

    # 模型和分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config(
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_positions=n_positions,
        n_ctx=n_positions,
        vocab_size=tokenizer.vocab_size,
    )
    model = GPT2LMHeadModel(config).cuda()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)  # DDP用于测试NCCL

    # 加载数据集
    def encode(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir="./dataset")
    dataset = dataset.map(encode, batched=True)
    dataset.set_format(type='torch', columns=['input_ids'])
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # 模型训练
    optimizer = AdamW(model.parameters(), lr=5e-5)
    torch.cuda.synchronize()
    start_time = time.time()
    model.train()
    total_steps = 0
    for epoch in range(epoch_count):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            total_steps += 1
            input_ids = batch['input_ids'].cuda()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if dist.get_rank() == 0:
                print(f"Epoch {epoch} | Step: {total_steps} | Loss: {loss.item()}")

    # 计算flops
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    tflops = 8 * FLOPS * total_steps * 2 / elapsed_time / 1e12  # 整个DDP的tflops
    if dist.get_rank() == 0:
        print(f"Training time: {elapsed_time:.2f} s")
        print(f"Total steps: {total_steps}")
        print(f"TFLOPS: {tflops:.2f}")


if __name__ == "__main__":
    main()
