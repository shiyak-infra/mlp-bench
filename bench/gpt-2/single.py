from datasets import load_dataset
from thop import profile
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

epoch_count = 10
batch_size = 32
n_embd = 2048
n_layer = 32
n_head = 16
n_positions = 1024


def main():
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
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # === thop 计算 FLOPs 和参数量 ===
    batch = next(iter(dataloader))
    input_ids = batch['input_ids'].cuda()
    # 只 forward 一次，不需要 label
    flops, params = profile(model, inputs=(input_ids,))
    print(f"FLOPs: {flops :.2f} FLOPs")
    print(f"Params: {params :.2f}")
    # ============================

    # 模型训练
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(epoch_count):
        for batch in dataloader:
            input_ids = batch['input_ids'].cuda()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch} | Loss: {loss.item()}")


if __name__ == "__main__":
    main()
