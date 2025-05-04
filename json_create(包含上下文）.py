import pandas as pd
import numpy as np
import json

# 读取数据集文件
df = pd.read_excel("dataset_test.xlsx",sheet_name="Sheet1")
INSTRUCTION = "根据上下文，该口语文本转换为正式的书面语文本"
# 构建json数据集，提取“原口语文本”列和“翻译后的书面语文本”列
json_data = []
for _, row in df.iterrows():
    content = {
        "instruction": INSTRUCTION,
        "input": {
            "context":row["口语文本所在上下文"],
            "text": row["原口语文本"]
        },
        "output": row["翻译后的书面语文本"]

    }
    json_data.append(content)

print(json_data)

# 导出文件，使用with open函数
with open("dataset_test.json","w",encoding="utf-8") as f:
    json.dump(json_data,f,ensure_ascii=False, indent=2)
