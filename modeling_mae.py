import torch 
import torch.nn as nn 
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block, PatchEmbed

from utils import count_parameters
from configuration import Config ,CifarConfig

cifarconfig = CifarConfig()
config = Config()
def random_masking(x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        to_keep = int(L * (1 - mask_ratio)) # patches to keep  based on  the mask ratio
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1] generate random noise 
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :to_keep] # keep only the unmasked 
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :to_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch = PatchEmbed(config.img_size, config.patch_size, config.in_chans, config.embed_dim)
        num_patches = config.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))  # Shape: (1, 1, 1024)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim), requires_grad=False)  
        # Shape: (1, num_patches + 1, 1024)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            Block(config.embed_dim, config.num_heads, config.mlp_ratio, qkv_bias=True, norm_layer=config.norm_layer)
            for _ in range(config.depth)
        ])
        
        # Normalization layer
        self.norm = config.norm_layer(config.embed_dim)

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch(x)  
        # Shape: (B, num_patches, embed_dim) = (B, num_patches, 1024)

        # Adding positional embeddings (excluding the cls_token)
        x = x + self.pos_embed[:, 1:, :]  
        # Shape: (B, num_patches, embed_dim) = (B, num_patches, 1024)

        # Random masking
        x, mask, ids_restore = random_masking(x, mask_ratio=config.mask_ratio)  
        # Shapes: x (B, L, 1024), mask (B, num_patches), ids_restore (B, num_patches)

        # Adding cls_token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  
        # Shape: (1, 1, 1024)
        
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  
        # Shape: (B, 1, 1024)

        # Concatenating cls_token to the sequence
        x = torch.cat((cls_tokens, x), dim=1)  
        # Shape: (B, L + 1, 1024)

        # Passing through the encoder blocks
        for blk in self.encoder_blocks:
            x = blk(x)  
        # Shape after each block: (B, L + 1, 1024)

        # Normalization
        x = self.norm(x)  
        # Shape: (B, L + 1, 1024)

        return x, mask, ids_restore


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Linear projection from encoder embedding dimension to decoder embedding dimension
        self.decoder_embed = nn.Linear(config.embed_dim, config.decoder_embed_dim, bias=True)  
        # Shape: (B, L, decoder_embed_dim) = (B, L, 512)

        # Mask token
        self.mask = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))  
        # Shape: (1, 1, 512)

        # Positional embedding for the decoder
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches + 1, config.decoder_embed_dim), requires_grad=False)  
        # Shape: (1, num_patches + 1, 512)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            Block(config.decoder_embed_dim, config.decoder_num_heads, config.mlp_ratio, qkv_bias=True, norm_layer=config.norm_layer)
            for _ in range(config.decoder_depth)
        ])
        
        # Normalization layer
        self.norm = config.norm_layer(config.decoder_embed_dim)

        # Output projection to pixel space
        self.out_linear = nn.Linear(config.decoder_embed_dim, config.patch_size**2 * config.in_chans, bias=True)  
        # Shape: (B, num_patches + 1, patch_size**2 * in_chans) = (B, num_patches + 1, 768)

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        
    def forward(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)  
        # Shape: (B, L, decoder_embed_dim) = (B, L, 512)

        # Create mask tokens and append them to the sequence
        mask_tokens = self.mask.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)  
        # Shape: (B, num_masked_patches, decoder_embed_dim) = (B, num_masked_patches, 512)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  
        # Shape: (B, num_patches, decoder_embed_dim) = (B, num_patches, 512)

        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  
        # Shape: (B, num_patches, decoder_embed_dim) = (B, num_patches, 512)

        # Append cls_token to the sequence
        x = torch.cat([x[:, :1, :], x_], dim=1)  
        # Shape: (B, num_patches + 1, decoder_embed_dim) = (B, num_patches + 1, 512)

        # Add positional embedding
        x = x + self.pos_embed  
        # Shape: (B, num_patches + 1, decoder_embed_dim) = (B, num_patches + 1, 512)

        # Passing through the decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)  
        # Shape after each block: (B, num_patches + 1, 512)

        # Normalization
        x = self.norm(x)  
        # Shape: (B, num_patches + 1, 512)

        # Projecting to pixel space
        x = self.out_linear(x)  
        # Shape: (B, num_patches + 1, patch_size**2 * in_chans) = (B, num_patches + 1, 768)

        # Removing the cls_token
        x = x[:, 1:, :]  
        # Final Shape: (B, num_patches, patch_size**2 * in_chans) = (B, num_patches, 768)

        return x


class MAE(nn.Module):
    def __init__(self, config):
        super(MAE, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config) 
    
    def forward(self, x):
        encoded_features, mask, ids_restore = self.encoder(x)
        out = self.decoder(encoded_features, ids_restore)
        
        return out,  mask , ids_restore

if __name__ == "__main__":
    model = MAE(cifarconfig)
    print(count_parameters(model))