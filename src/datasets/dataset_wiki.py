import os.path

import torch
from src.bpe import ByteBPE

class WikiDataset:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.pad_idx = -1  

def make_wiki_dataset(
        data_dir,
        device,
        context_size,
        vocab_size
):
    train = open(os.path.join(data_dir, "wiki.train.raw"), "r", encoding="utf-8").read()
    dev = open(os.path.join(data_dir, "wiki.valid.raw"), "r", encoding="utf-8").read()
    test = open(os.path.join(data_dir, "wiki.test.raw"), "r", encoding="utf-8").read()

    split = {"train": train, "dev": dev, "test": test}

    BPE = ByteBPE()

    BPE.train(text=train, vocab_size=vocab_size)

    def make_chunks(ids, context_size):
        for i in range(0, len(ids), context_size):
            if len(ids[i:i + context_size+1]) < context_size+1:
                continue
            yield ids[i:i + context_size+1]

    encoded_split = {}
    for name, text in split.items():
        ids = BPE.encode(text)
        chunks = list(make_chunks(ids, context_size))

        inputs  = [chunk[:-1] for chunk in chunks]
        targets = [chunk[1:] for chunk in chunks]

        inputs  = torch.tensor(inputs, dtype=torch.long, device=device)
        targets = torch.tensor(targets, dtype=torch.long, device=device)

        encoded_split[name] = WikiDataset(inputs, targets)

    encoded_split["vocab_size"] = vocab_size

    return encoded_split