import torch
print(torch.__version__)        # should show e.g. "2.x.x+cu121"
print(torch.cuda.is_available()) # should be True
print(torch.cuda.get_device_name(0))  # shows your GPU name
