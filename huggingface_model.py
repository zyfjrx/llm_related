# Load model directly
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("/home/bmh/project/model/AI-ModelScope/siglip-base-patch16-224")
print(model)