#模型下载
from modelscope import snapshot_download
cache_dir = "/home/bmh/project/model"
model_dir = snapshot_download('AI-ModelScope/siglip-base-patch16-224',cache_dir=cache_dir)