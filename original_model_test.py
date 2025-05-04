from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import jieba

model_path = '/root/autodl-tmp/Qwen/Qwen2___5-Coder-7B-Instruct'
file = 'dataset_test.json'

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, device_map = "auto", torch_dtype = torch.bfloat16)

def json_to_dict(file_path):
    try:
        with open(file_path, 'r',encoding='utf-8') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
    except json.JSONDecodeError:
        print("错误：JSON格式不正确")
    except Exception as e:
        print(f"未知错误：{str(e)}")
    return None

# 构建中文分词
class chinesetokenizer:
    def tokenize(self, text):
        return list(jieba.cut(text))


# 构建评估函数
def evaluate(references, predictions):
    smooth = SmoothingFunction().method1
    bleu_scores = []
    rouge_scores = []

    scorer = rouge_scorer.RougeScorer(['rougeL'],
                                      use_stemmer=False,
                                      tokenizer=chinesetokenizer())  # 使用jieba分词

    for ref, pred in zip(references, predictions):
        # BLEU计算保持不变
        ref_tokens = [list(jieba.cut(ref))]
        pred_tokens = list(jieba.cut(pred))
        bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)
        bleu_scores.append(bleu)

        # ROUGE计算
        scores = scorer.score(ref, pred)
        rouge_scores.append(scores['rougeL'].fmeasure)

    return {
        'bleu': sum(bleu_scores) / len(bleu_scores),
        'rouge': sum(rouge_scores) / len(rouge_scores)
    }

if __name__ == "__main__":
    file_path = "/root/dataset_test.json"  # 替换为你的JSON文件路径
    result = json_to_dict(file_path)

predictions = []
references = []

# 进行测试
for i in range(len(result)):
    inputs = tokenizer.apply_chat_template(
        [{"role":"user", "content":"你现在是一个专业的口语翻译机,根据句子所在上下文，请讲该口语文本转成书面语文本。"},
        {"role":"user", "content":result[i]['input']['context']},
        {"role":"user", "content":result[i]['input']['text']}],
        add_generation_prompt = True,
        tokenize = True,
        return_tensors = "pt",
        return_dict= True
    ).to('cuda')
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        references.append(result[i]['output'])
        predictions.append(decoded)

results = evaluate(references, predictions)
print(f"BLEU Score: {results['bleu']:.4f}")
print(f"ROUGE-L F1 Score: {results['rouge']:.4f}")