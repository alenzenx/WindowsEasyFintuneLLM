import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. 載入「原始base模型」(與你微調時對應的版本要一致)
base_model_path = "Llama-2-7b-hf"  # 或其他 base model 目錄
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,  # 若顯卡支援FP16
    device_map="cuda:0",         # 讓HF分配到GPU上跑
)

# 2. 載入 LoRA Adapter 權重
adapter_path = "qlora_output/epoch_0"  # 這裡放置 adapter_model.safetensors 等檔案的位置
model = PeftModel.from_pretrained(model, adapter_path)

# 3. 進行推理
prompt = "你好，我想問你一個問題："
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
