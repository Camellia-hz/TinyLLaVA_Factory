import os

import torch
import torch.nn as nn
from safetensors.torch import load_file

class Connector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self._connector = None

    def load_model(self, **kwargs):
        pretrained_connector_path = kwargs.get('pretrained_connector_path', None)
        if pretrained_connector_path is not None:
            # pretrained_connector_path = os.path.join(pretrained_connector_path, 'pytorch_model.bin')
            # connector_weights = torch.load(pretrained_connector_path, map_location='cpu')
            connector_weights = load_file("/mnt/csi-data-aly/user/haozhou/Projects/TinyLLaVA_Factory/checkpoints/Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP/model.safetensors")
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            # import pdb;pdb.set_trace()
            self._connector.load_state_dict(get_w(connector_weights, '_connector'))
            print(f'Loading connector from {pretrained_connector_path}...')

        for p in self._connector.parameters():
            p.requires_grad = False
   

    
    
    def forward(self, x):
        return self._connector(x)
        

  
