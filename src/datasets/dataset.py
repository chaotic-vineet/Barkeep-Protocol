import torch
import random

def make_dataset(file_path, device, context_size=10, seed=42):
    accelerator = device

    data = open(file_path, "r").read().splitlines()
    names = [name for name in data if len(name) <= context_size - 1]

    letters = sorted(set("".join(names)))
    ltr_to_idx = {ltr: idx for idx, ltr in enumerate(letters, start=1)}
    ltr_to_idx["."] = 0
    ltr_to_idx["#"] = len(ltr_to_idx)

    vocab_size = len(ltr_to_idx)
    idx_to_ltr = {v: k for k, v in ltr_to_idx.items()}
    pad_idx = ltr_to_idx["#"]

    class Dataset:
        def __init__(self, words):
            Inputs, Targets, Lengths = [], [], []
            for word in words:
                complete = [0] + [ltr_to_idx[ltr] for ltr in word] + [0]

                Input, Target = complete[:-1], complete[1:]
                Length = len(Input)

                pad = [pad_idx] * (context_size - Length)

                Inputs.append(Input + pad)
                Targets.append(Target + pad)
                Lengths.append(Length)

            self.inputs  = torch.tensor(Inputs,  dtype=torch.long, device=accelerator)
            self.targets = torch.tensor(Targets, dtype=torch.long, device=accelerator)
            self.lengths = torch.tensor(Lengths, dtype=torch.long, device=accelerator)
            self.pad_idx = pad_idx

        def __len__(self):
            return len(self.inputs)

        def stats(self):
            n = len(self)
            real = self.lengths
            return {
                "examples":     n,
                "mean_len":     real.float().mean().item(),
                "max_len":      real.max().item(),
                "min_len":      real.min().item(),
                "real_tokens":  real.sum().item(),
                "pad_tokens":   n * context_size - real.sum().item(),
                "pad_fraction": 1 - real.sum().item() / (n * context_size),
            }

    random.seed(seed)
    random.shuffle(names)
    n1, n2 = int(0.8 * len(names)), int(0.9 * len(names))

    return {
        "trn":         Dataset(names[:n1]),
        "dev":         Dataset(names[n1:n2]),
        "test":        Dataset(names[n2:]),
        "vocab_size":  vocab_size,
        "ltr_to_idx":  ltr_to_idx,
        "idx_to_ltr":  idx_to_ltr
    }