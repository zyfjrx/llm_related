import json

def replace_optimized_thought(json_data):
    """
    将JSON数据中的"**Optimized Chain of Thought:**  \n\n"替换为单引号
    """
    for item in json_data:
        if 'output' in item:
            # 替换字符串

            item['output'] = item['output'].replace("**Optimized Chain of Thought:**  \n\n", "")
            item['output'] = item['output'].replace("**Optimized Chain of Thought:**  \n", "")
            item['output'] = item['output'].replace("**Optimized Chain of Thought:**", "")
            item['output'] = item['output'].replace("**Answer:**", "")
            item['output'] = item['output'].replace("**Answer:**  \n", "")
    return json_data


# 处理JSON数据
with open("/Users/zhangyf/Documents/宠物数据/datasets/Food & Diet/datasets-4_AjgzGtdvgO-alpaca-2025-05-26.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
processed_data = replace_optimized_thought(json_data)

# 打印处理后的数据
# print(json.dumps(processed_data, indent=2))

# 可选：将处理后的数据保存到文件
with open('processed_data.json', 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)