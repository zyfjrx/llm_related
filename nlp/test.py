from swift.llm import load_dataset
# 支持重采样：（超过108后进行重采样）
dataset = load_dataset(['swift/self-cognition#500'], model_name=['小黄', 'Xiao Huang'], model_author=['魔搭', 'ModelScope'])[0]
print(dataset)
"""
Dataset({
    features: ['messages'],
    num_rows: 500
})
"""