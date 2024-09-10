import torch 



class CifarConfig:
    def __init__(self,
                 img_size=32,         # CIFAR-10 image size
                 patch_size=4,        # Patch size
                 in_chans=3,          # Number of input channels (RGB)
                 embed_dim=128,       # Reduced embedding dimension
                 decoder_embed_dim=64, # Reduced decoder embedding dimension
                 depth=4,             # Reduce depth since CIFAR-10 is simpler
                 decoder_depth=2,     # Reduce decoder depth
                 num_heads=4,
                 decoder_num_heads=4,
                 mlp_ratio=2.,
                 norm_layer=torch.nn.LayerNorm,
                 norm_pix_loss=False,
                 mask_ratio=0.75):
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.depth = depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.norm_pix_loss = norm_pix_loss
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2  # Number of patches

class Config:
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=256,
                 decoder_embed_dim=128,
                 depth=8,
                 decoder_depth=4,
                 num_heads=4,
                 decoder_num_heads=4,
                 mlp_ratio=2.,
                 norm_layer=torch.nn.LayerNorm,
                 norm_pix_loss=False,
                 mask_ratio=0.75):
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.depth = depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.norm_pix_loss = norm_pix_loss
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2

