## 最終成果 : 加速LLM訓練及推理
Windows 11 + CUDA 12.4 + cuDNN v8.9.7 + python 3.11.0 + pytorch 2.6.0+cu124 + Triton 3.2.0.post12

# 自行安裝python 3.11.0 + CUDA 12.4 + cuDNN v8.9.7

# 增加Windows系統環境變數(下面的路徑是預設路徑)
CUDA_PATH
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4

# 下載Visual Stduio Installer
# 下載安裝Visual Studio Build Tools 2022    version:17.13.2
# 下載安裝 MSVC四件套
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