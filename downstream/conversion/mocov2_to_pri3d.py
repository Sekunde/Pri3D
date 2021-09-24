import torch
import sys

if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2]

    model = torch.load(input, map_location="cpu") 
    new_model = {}
    for key in model['state_dict'].keys():
        if 'module.encoder_q' not in key:
            continue
        print(key)
        new_key = key.replace('module.encoder_q', 'backbone')
        weight = model['state_dict'][key]
        new_model[new_key] = weight

    torch.save(new_model, output)
