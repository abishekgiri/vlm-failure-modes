import torch
import torch.nn.functional as F

def token_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    return entropy.mean().item()
