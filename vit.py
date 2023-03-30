import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, image_res=(28, 28), n_patches=7):
        super(ViT, self).__init__()
        self.n_patches = n_patches
        self.height = image_res[0]
        self.width = image_res[1]

        assert self.height % n_patches == 0, 'Height not cleanly divided!'
        assert self.width % n_patches == 0, 'Width not cleanly divided!'

        self.linear = nn.Linear()

    def patchify(self, images, n_patches):
        num_images, channel, height, width = images.shape

        assert height == width, 'Implemented only for square images for now!'
        
        # Gives Batch x Number of Patches x Number of Pixels in a Patch
        patches = torch.zeros(num_images, n_patches ** 2, height * width * channel//n_patches ** 2) 
        patch_size = height // n_patches

        # Iterating through each image in a batch
        for idx, image in enumerate(images):
            for y_patch in range(n_patches):
                for x_patch in range(n_patches):
                    patch = image[:, y_patch * patch_size:(y_patch + 1) * patch_size, x_patch * patch_size:(x_patch + 1) * patch_size]
                    patches[idx, y_patch * n_patches + x_patch] = patch.flatten()
        
        return patches

    def forward(self, images):
        patches = self.patchify(images, self.n_patches)
        return patches

if __name__ == '__main__':
    model = ViT(image_res=(28, 28), n_patches=7)
    x = torch.randn(24, 1, 28, 28)
    output = model(x)
    print(output.shape)