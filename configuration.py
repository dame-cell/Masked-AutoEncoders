import torch 

class MAEConfig:
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75):
        self.image_size = image_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.encoder_layer = encoder_layer
        self.encoder_head = encoder_head
        self.decoder_layer = decoder_layer
        self.decoder_head = decoder_head
        self.mask_ratio = mask_ratio