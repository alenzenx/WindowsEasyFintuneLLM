# my_dummy_dataset.py
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
