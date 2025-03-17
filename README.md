# Windows Easy Fintune LLM
![image](https://github.com/alenzenx/WindowsEasyFintuneLLM/blob/main/goodjob.jpg)

## **Software Configuration**

**OS : Windows 11**

**CUDA : 12.4**

**cuDNN : v8.9.7**

**python : 3.11.0**

**pytorch : 2.6.0+cu124**

**triton-windows : 3.2.0.post12**

**torchtune : 0.5.0+cu124**

**LLM : LLaMA-2-7B** *(Used in this tutorial)*

**Finetune method : QLoRA** *(Used in this tutorial)*

## **Hardware Configuration**

**GPU : NVIDIA GeForce RTX 3060 12GB** *(important : 12GB of VRAM is the minimum standard.)*

**RAM : 16GB** *(Better up)*

## **1. Download and Install**
### *python 3.11.0 + CUDA 12.4 + cuDNN v8.9.7*

#### CUDA and cuDNN similar install tutorial
https://medium.com/@alenzenx/安裝-cuda12-6-與-cudnn-8-9-7-34f95ef8ce7f

#### Verify CUDA GPU execution
    python GPUtest.py

## **2. Install MSVC for Triton-Windows**
#### Add a Windows System variables (The following path is the default path)
Variable name
```
CUDA_PATH
```    
Variable value
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
```

#### Download Visual Stduio Installer
#### Open Visual Stduio Installer
#### Install Visual Studio Build Tools 2022 version : 17.13.2
#### Install MSVC (Select within Visual Studio Build Tools 2022 version)
##### Click "Modify" 
##### Click "Individual components"
##### Select
1. MSVC v143 - VS 2022 C++ x64/x86 build tools(Latest)

2. Windows 11 SDK(10.0.22621.0)

3. C++ CMake tools for Windows

4. MSBuild support for LLVM(clang-cl) toolset 

#### Add the following path to Path under Windows User variables for User (The following path is the default path)
```
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64
```

#### Verify Triton execution
```
python test_triton.py
```

## **3. Create Virtual Environment**
#### (Change "C:\Users\User\Desktop\ourllm" to your path)
```
python -m venv C:\Users\User\Desktop\ourllm
```
```
C:\Users\User\Desktop\ourllm\Scripts\Activate.ps1
```
## **4. Install requirements**
    pip install -r requirements.txt

## **5. Download raw LLaMA-2-7B**
```
pip install llama-stack
```
```
llama model list --show-all
```
```
llama download --source meta --model-id Llama-2-7b
```

#### LLaMA-2 verification key (a URL starting with https)
    https://download.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiMWh0d3JyeWVxOXE1cWpjMTQ5aDQ2OWx5IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZG93bmxvYWQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0MTk1ODM1MH19fV19&Signature=nCSq%7ECseY3cvvI5w7THDAAXAvaiqP81ibq5nLCztW1efQmL-f67TvxGrblYUGV5Kg7URAsDxJNp5NFdOVoyOX5E5fpFm1Dzi2xAfsrunyGVnud-uliH8HdHoEwT9Pmin5qSt4slG9v2n4hSw7t-htP4dd5yh69rpf7GJWH02QKc66Axf4%7EoQ1AhFc0cLpSpS3MUMDp7D1m2jEjT98J4Ee3Hj1eH%7EtU0mGytyncEb-W1bNEZt8TdTIDwE8pY2S9sXpzGkbQrHv5A4QvR0fqEcvio47uvVjYqSH7ExCHJP5WeYEuT6lXNFgfn59oe0coyliIseAXLQet7X7Jbh2m64Tw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=587287740993120

## **6. Convert the raw LLaMA-2 model into hf format (hf format=huggingface format)**
```
pip install protobuf sentencepiece
```
```
python convert_llama_weights_to_hf.py --input_dir "Llama-2-7b" --model_size 7B --output_dir "Llama-2-7b-hf" --llama_version 2
```

## **7. Fine-tune LLaMA-2 using Torchtune.**
#### torchtune directory
    tune ls

#### Copy the default QLoRA single-GPU training config file from tune
    tune cp llama2/7B_qlora_single_device custom_config.yaml

### **Write custom_config.yaml**
#### Change output_dir path
    output_dir: qlora_output

#### Change tokenizer path
    tokenizer:
      _component_: torchtune.models.llama2.llama2_tokenizer
      path: Llama-2-7b-hf/tokenizer.model
      max_seq_len: null

#### Change checkpointer path and Change to save only QLoRA weights
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

#### Floating-point format : bf16 -> fp32 (Geforce GPU need)
    dtype: fp32

#### Change batch size
    dataset:
      _component_: torchtune.datasets.alpaca_cleaned_dataset
      packed: False  # True increases speed
    seed: null
    shuffle: True
    batch_size: 4

#### Verify custom_config.yaml
    tune validate custom_config.yaml

## **If you want to train the full version instead of a dummy test : proceed to Step 9.**

## **8. Create Dummy Test**
#### Change batch size and create dummy test path
    dataset:
      _component_: my_dummy_dataset.MyDummyDataset
      data_file: "./dummy_alpaca.json"
      packed: False  # True increases speed
    seed: null
    shuffle: True
    batch_size: 4

#### Verify custom_config.yaml
    tune validate custom_config.yaml

#### Create dummy_alpaca.json
    [
        {
        "instruction": "Dummy Instruction",
        "input": "Dummy Input",
        "output": "Dummy Output"
        }
    ]

#### Create my_dummy_dataset.py
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

## **9. Single GPU Finetune LLM (Train)**
    tune run lora_finetune_single_device --config custom_config.yaml

## **10. Inference**
    python test.py