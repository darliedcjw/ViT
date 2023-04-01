import torch
import torch.nn as nn
import numpy as np

class ViT(nn.Module):
    def __init__(self, image_res=(1, 28, 28), n_patches=7, hidden_d=8):
        super(ViT, self).__init__()
        self.channel = image_res[0]
        self.height = image_res[1]
        self.width = image_res[2]
        self.n_patches = n_patches
        self.hidden_d = hidden_d

        assert self.height % n_patches == 0, 'Height not cleanly divided!'
        assert self.width % n_patches == 0, 'Width not cleanly divided!'
 
        self.patch_size = self.height // self.n_patches, self.width // self.n_patches

        # Linear Mapping (Flatten)
        self.linear_input = self.patch_size[0] * self.patch_size[1] * self.channel
        self.linear = nn.Linear(self.linear_input, self.hidden_d)

        # Layer Normalization (Normalise based on the last dimension)
        '''
        Batchnorm2d calculates mean and sd with respect to each channel for the batch
        LayerNorm calculates mean and sd with respect to last D dimensions of normalized shape
        '''
        self.ln = nn.LayerNorm(hidden_d)

        # Learnable Classification Token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.tensor(self.get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d)))
        self.pos_embed.requires_grad = False        

    def patchify(self, images, n_patches):
        num_images, _, _, _ = images.shape

        assert self.height == self.width, 'Implemented only for square images for now!'
        
        # Gives Batch x Number of Patches x Number of Pixels in a Patch
        patches = torch.zeros(num_images, n_patches ** 2, self.height * self.width * self.channel//n_patches ** 2) 

        # Iterating through each image in a batch
        for idx, image in enumerate(images):
            for y_patch in range(n_patches):
                for x_patch in range(n_patches):
                    patch = image[:, y_patch * self.patch_size[0]:(y_patch + 1) * self.patch_size[0], x_patch * self.patch_size[1]:(x_patch + 1) * self.patch_size[1]]
                    patches[idx, y_patch * n_patches + x_patch] = patch.flatten()
        
        return patches
    
    # Postional Function
    '''
    For each patch, for each pixel, get position
    '''
    def get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result

    def forward(self, images):
        patches = self.patchify(images, self.n_patches)
        tokens = self.linear(patches)
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        pos_emb = self.pos_embed.repeat(images.shape[0], 1, 1)
        out = tokens + pos_emb
        out = self.ln(out)
        return out
        # return tokens

if __name__ == '__main__':
    model = ViT(image_res=(3, 28, 28), n_patches=7)
    x = torch.randn(24, 3, 28, 28)
    output = model(x)
    print(output.shape)