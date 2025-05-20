import argparse
import os
import time

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling

import utils

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=2, type=int, help="Epoch of training.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size of training.")
parser.add_argument("--max_length", default=128, type=int, help="Sequence length of dataset.")
args = parser.parse_args()

# 验证环境
if not (torch.cuda.is_available() and torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32):
    raise ValueError("Not available")


def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# 加载分词器和模型
local_rank = setup_distributed()
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', cache_dir='../tokenizer')
model = BertForMaskedLM.from_pretrained('bert-large-uncased', cache_dir='../model').cuda()
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# 准备数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir="../dataset")


# 文本分词和编码
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length)


tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 构造 MLM 数据
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
sampler = DistributedSampler(dataset)
dataloader = DataLoader(tokenized_datasets, batch_size=args.batch_size, collate_fn=data_collator, sampler=sampler)

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
torch.cuda.synchronize()
start_time = time.time()
model.train()
total_steps = 0
for epoch in range(args.epoch):
    total_loss = 0
    for batch in dataloader:
        total_steps += 1
        outputs = model(**{k: v.cuda() for k, v in batch.items()})
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        if dist.get_rank() == 0:
            print(f"Epoch {epoch} | Step: {total_steps} | Loss: {loss.item()}")

# 计算flops
torch.cuda.synchronize()
end_time = time.time()
elapsed_time = end_time - start_time
if dist.get_rank() == 0:
    tflops = torch.distributed.get_world_size() * utils.get_flo(args.batch_size,
                                                                args.max_length) * total_steps * 2 / elapsed_time / 1e12  # 整个DDP的tflops
    print(f"Training time: {elapsed_time:.2f} s")
    print(f"Total steps: {total_steps}")
    print(
        f"Throughput: {torch.distributed.get_world_size() * total_steps * args.batch_size / elapsed_time:.2f} sequences/sec")
    print(f"Tflops: {tflops:.2f}")
