# Windows Easy Fintune LLM
![image](https://github.com/alenzenx/WindowsEasyFintuneLLM/blob/main/goodjob.jpg)

**System : Windows 11**

**CUDA : 12.4**

**cuDNN : v8.9.7**

**python : 3.11.0**

**pytorch : 2.6.0+cu124**

**Triton : 3.2.0.post12**

## **1. Download and Install**
python 3.11.0 + CUDA 12.4 + cuDNN v8.9.7

## **2. Install C++ Compiler for Triton-Windows**
#### 增加Windows系統環境變數(下面的路徑是預設路徑)
CUDA_PATH

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4

#### Download Visual Stduio Installer
#### Install Visual Studio Build Tools 2022 version : 17.13.2
#### Install MSVC
1. MSVC v143 - VS 2022 C++ x64/x86 build tools(Latest)
2. Windows 11 SDK(10.0.22621.0)
3. C++ CMake tools for Windows
4. MSBuild support for LLVM(clang-cl) toolset 

# 修改Windows User環境變數Path項，加入(下面的路徑是預設路徑)
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64

# 虛擬環境(下面的路徑 C:\Users\User\Desktop\ourllm 自己修改，且 \Scripts\Activate.ps1 那行要一起改呦)
python -m venv C:\Users\User\Desktop\ourllm
C:\Users\User\Desktop\ourllm\Scripts\Activate.ps1

# 安裝
pip install -r requirements.txt

# 下載原始llama2
pip install llama-stack
llama model list --show-all
llama download --source meta --model-id Llama-2-7b

# 我的 llama2 驗證金鑰(一串https的網址)
https://download.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiMWh0d3JyeWVxOXE1cWpjMTQ5aDQ2OWx5IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZG93bmxvYWQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0MTk1ODM1MH19fV19&Signature=nCSq%7ECseY3cvvI5w7THDAAXAvaiqP81ibq5nLCztW1efQmL-f67TvxGrblYUGV5Kg7URAsDxJNp5NFdOVoyOX5E5fpFm1Dzi2xAfsrunyGVnud-uliH8HdHoEwT9Pmin5qSt4slG9v2n4hSw7t-htP4dd5yh69rpf7GJWH02QKc66Axf4%7EoQ1AhFc0cLpSpS3MUMDp7D1m2jEjT98J4Ee3Hj1eH%7EtU0mGytyncEb-W1bNEZt8TdTIDwE8pY2S9sXpzGkbQrHv5A4QvR0fqEcvio47uvVjYqSH7ExCHJP5WeYEuT6lXNFgfn59oe0coyliIseAXLQet7X7Jbh2m64Tw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=587287740993120

# 轉換原始llama2成hf格式
下載 https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
pip install protobuf sentencepiece
python convert_llama_weights_to_hf.py --input_dir "Llama-2-7b" --model_size 7B --output_dir "Llama-2-7b-hf" --llama_version 2

# tune的目錄
tune ls

# 複製tune預設的QLoRA單GPU訓練的Config檔案
tune cp llama2/7B_qlora_single_device custom_config.yaml

# custom_config.yaml
## 更改output_dir path
output_dir: qlora_output

## 更改tokenizer path
tokenizer:
  _component_: torchtune.models.llama2.llama2_tokenizer
  path: Llama-2-7b-hf/tokenizer.model
  max_seq_len: null

## 更改checkpointer path 跟 改成只存QLoRA權重
checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: Llama-2-7b-hf
  checkpoint_files: [
    model-00001-of-00003.safetensors,
    model-00002-of-00003.safetensors,
    model-00003-of-00003.safetensors
  ]
  adapter_checkpoint: null
  recipe_checkpoint: null
  output_dir: ${output_dir}
  model_type: LLAMA2
resume_from_checkpoint: False
save_adapter_weights_only: True

## 更改batch size 及 建立dummy的路徑
dataset:
  _component_: my_dummy_dataset.MyDummyDataset
  data_file: "./dummy_alpaca.json"
  packed: False  # True increases speed
seed: null
shuffle: True
batch_size: 4

## bf16改fp32
dtype: fp32

# 驗證custom_config.yaml的編寫格式
tune validate custom_config.yaml

# 建立dummy
建立 dummy_alpaca.json
放入
[
    {
      "instruction": "Dummy Instruction",
      "input": "Dummy Input",
      "output": "Dummy Output"
    }
]

建立 my_dummy_dataset.py
放入
import json
from torch.utils.data import Dataset

class MyDummyDataset(Dataset):
    def __init__(self, tokenizer=None, data_file=None, packed=False):
        """
        tokenizer: 由 Torchtune 以位置引數 (positional arg) 傳入
        data_file: YAML 中指定的關鍵字引數
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.packed = packed  # <- 關鍵字參數接收 "packed"

        # 如果 data_file 有指定，就載入資料
        if data_file is not None:
            with open(data_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            # 沒有提供檔案就給個空 list
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return {
            "instruction": example.get("instruction", ""),
            "input": example.get("input", ""),
            "output": example.get("output", ""),
        }

# LoRA單GPU訓練
tune run lora_finetune_single_device --config custom_config.yaml

# 推理
python test.py