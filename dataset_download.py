from datasets import load_dataset,load_from_disk
# import shutil
# import os
# cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
# if os.path.exists(cache_dir):
#     shutil.rmtree(cache_dir)

# 下载数据集到本地缓存（默认路径为 ~/.cache/huggingface/datasets）
# dataset = load_from_disk("/home/bmh/.cache/huggingface/hub/datasets--liuhaotian--LLaVA-CC3M-Pretrain-595K/snapshots/814894e93db9e12a1dee78b9669e20e8606fd590")
dataset = load_dataset("liuhaotian/LLaVA-CC3M-Pretrain-595K")
print(dataset)
# dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K")

# 指定本地保存路径（可选）
# dataset.save_to_disk("/home/bmh/project/dataset")  # 保存为本地目录

