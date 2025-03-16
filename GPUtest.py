# pytorch test
import torch
flag = torch.cuda.is_available()
if flag:
    # print("CUDA available")
    ngpu= 1
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # print("GPU model: ",torch.cuda.get_device_name(0))
    print(device)
    print("pytorch can use GPU")
else:
    # print("CUDA unavailable")
    print("pytorch can't use GPU")