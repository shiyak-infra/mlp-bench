import argparse
import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling

import utils

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=2, type=int, help="Epoch of training.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size of training.")
parser.add_argument("--max_length", default=128, type=int, help="Sequence length of dataset.")
args = parser.parse_args()

# 验证环境
if not (torch.cuda.is_available() and torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32):
    raise ValueError("Not available")

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', cache_dir='../tokenizer')
model = BertForMaskedLM.from_pretrained('bert-large-uncased', cache_dir='../model').cuda()

# 准备数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir="../dataset")


# 文本分词和编码
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length)


tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 构造MLM数据
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
dataloader = DataLoader(tokenized_datasets, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
torch.cuda.synchronize()
start_time = time.time()
model.train()
total_steps = 0
for epoch in range(args.epoch):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        total_steps += 1
        outputs = model(**{k: v.cuda() for k, v in batch.items()})
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1} | loss: {total_loss / len(dataloader):.4f}")

# 计算flops
torch.cuda.synchronize()
end_time = time.time()
elapsed_time = end_time - start_time
tflops = utils.get_flo(args.batch_size, args.max_length) * total_steps * 2 / elapsed_time / 1e12  # 整个DDP的tflops
print(f"Training time: {elapsed_time:.2f} s")
print(f"Total steps: {total_steps}")
print(f"Throughput: {total_steps * args.batch_size / elapsed_time:.2f} sequences/sec")
print(f"Tflops: {tflops:.2f}")
