 
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

class VectorQuantizer(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self.gamma = 0.99
        self.gamma_decay = 0.999
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._ema_embedding = nn.Parameter(self._embedding.weight.data.clone(), requires_grad=False)
    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = rearrange(inputs, 'b c h w -> b h w c')
        input_shape = inputs.shape
        # Flatten input
        flat_input = inputs.reshape(-1, self._embedding_dim)
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='none') 
        q_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction='none')
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        # convert quantized from BHWC -> BCHW
        quantized = rearrange(quantized, 'b h w c -> b c h w')

        return quantized, loss, encoding_indices.view(input_shape[:-1])
    
    def decode(self, indices):
        # convert indices from BCHW -> BHWC
        flat_indices = indices.contiguous().view(-1, 1)

        # One hot
        encodings = torch.zeros(flat_indices.shape[0], self._num_embeddings, device=indices.device)

        # Use scatter to get one hot encodings
        encodings.scatter_(1, flat_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(indices.shape + (self._embedding_dim,))

        # convert quantized from BHWC -> BCHW
        quantized = rearrange(quantized, 'b h w c -> b c h w')

        return quantized

    
class VQVAE(nn.Module):
    def __init__(self, inchan, scaling, num_embeddings, embedding_size):
        super(VQVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(inchan, scaling, 3, stride=1, padding=1), # Scaling x 256 x 256
            nn.SiLU(),
            nn.Conv2d(scaling, scaling*2, 3, stride=2, padding=1), # Scaling*2 x 128 x 128
            nn.SiLU(),
            nn.Conv2d(scaling*2, scaling*4, 3, stride=2, padding=1), # Scaling*4 x 64 x 64
            nn.SiLU(),
            nn.Conv2d(scaling*4, scaling*8, 3, stride=2, padding=1), # Scaling*8 x 32 x 32
            # nn.SiLU(),
            # nn.Conv2d(scaling*8, scaling*16, 3, stride=2, padding=1), # Scaling*16 x 16 x 16
            # nn.SiLU(),
            # nn.Conv2d(scaling*16, scaling*32, 3, stride=2, padding=1), # Scaling*32 x 8 x 8
            # nn.SiLU(),
            # nn.Conv2d(scaling*32, scaling*64, 3, stride=2, padding=1), # Scaling*64 x 4 x 4
        )

        self.prequantize = nn.Conv2d(scaling*8, embedding_size, 1, stride=1, padding=0)
        self.quantize = VectorQuantizer(num_embeddings, embedding_size, 0.25)
        self.postquantize = nn.Conv2d(embedding_size, scaling*8, 1, stride=1, padding=0)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(scaling*8, scaling*8, 3, stride=1, padding=1),

            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(scaling*8, scaling*4, 3, stride=1, padding=1),

            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(scaling*4, scaling*2, 3, stride=1, padding=1),

            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(scaling*2, scaling, 3, stride=1, padding=1),
        )

        self.out = nn.Conv2d(scaling, inchan, 3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.prequantize(x)
        x, loss, _ = self.quantize(x)
        x = self.postquantize(x)
        x = self.decoder(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x, loss
    
    def encode(self, x):
        x_ = x
        if x.ndim == 5:
            x = rearrange(x, 'b c h w t -> (b t) c h w')
        x = self.encoder(x)
        x = self.prequantize(x)
        x, _, indices = self.quantize(x)

        if x_.ndim == 5:
            indices = rearrange(indices, '(b t) h w -> b h w t', t=x_.shape[-1])
        return indices
    
    def decode(self, indices):
        x_ = indices
        if indices.ndim == 4:
            indices = rearrange(indices, 'b h w t -> (b t) h w')
        y = self.quantize.decode(indices)
        y = self.postquantize(y)
        y = self.decoder(y)
        y = self.out(y)
        y = torch.sigmoid(y)

        if x_.ndim == 4:
            y = rearrange(y, '(b t) c h w -> b c h w t', t=x_.shape[-1])
        
        return y

class Discriminator(nn.Module):
    # Simple discriminator

    def __init__(self, inchan, scaling):
        super(Discriminator, self).__init__()

        self.blocks = nn.Sequential(
            nn.Conv2d(inchan, scaling, 3, stride=1, padding=1), # Scaling x 256 x 256
            nn.SiLU(),
            nn.Conv2d(scaling, scaling*2, 3, stride=2, padding=1), # Scaling*2 x 128 x 128
            nn.SiLU(),
            nn.Conv2d(scaling*2, scaling*4, 3, stride=2, padding=1), # Scaling*4 x 64 x 64
            nn.SiLU(),
            nn.Conv2d(scaling*4, scaling*8, 3, stride=2, padding=1), # Scaling*8 x 32 x 32
            nn.SiLU(),
            nn.Conv2d(scaling*8, 1, 3, stride=1, padding=1), # 1 x 32 x 32
        )
    def forward(self, x):
        return self.blocks(x)