#!/usr/bin/env python

import pickle as pkl
import sys
import torch


if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2]

    obj = torch.load(input, map_location="cpu")

    newmodel = {}

    for key in obj['model']:
        if key.startswith('backbone.encoder.'):
            new_key = key.replace('backbone.encoder.','')
            weight = obj['model'][key]
            newmodel[new_key] = weight

    torch.save(newmodel, output)
