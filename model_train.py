from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model

def process_func(example):
    max_len = 384
    input_ids = []
    attention_mask = []
    labels = []
    instruction = tokenizer(f"<|im_start|>system\n现在你要进行口语文本的翻译，根据句子所处上下文内容，将该口语文本翻译为书面语文本<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens = False)
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"]+response["input_ids"]+[tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"]+response["attention_mask"]+[1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > max_len:  # 做一个截断
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        labels = labels[:max_len]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenizer =AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen/Qwen2___5-Coder-7B-Instruct/', use_fast = False, trust_remote_code = True)
model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/Qwen/Qwen2___5-Coder-7B-Instruct/', device_map = "auto", torch_dtype = torch.bfloat16)

df = pd.read_json('/root/dataset_context.json')
ds = Dataset.from_pandas(df)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32,
    lora_dropout=0.1# Dropout 比例
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir="./output/Qwen2.5_instruct_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()