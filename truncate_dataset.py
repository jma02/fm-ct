import torch
d = torch.load("dataset.pt") 

indices = torch.arange(0, len(d), 100)

d = d[indices]
torch.save(d, "dataset-lite.pt")

