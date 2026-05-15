import torch
import torch.nn as nn
import torch.nn.functional as F
from .api import DepthAnything3
import glob
import os

    
class DAV3Module(nn.Module):
    def __init__(self, config=None):
        super(DAV3Module, self).__init__()
        
        self.model = DepthAnything3.from_pretrained("depth-anything/da3-base", )
        
        

    def inference_images(self, images):
        # images: List of image paths
        return self.model.inference(images)
    
    def extract_features(self, image_tensor,image_mask = None):
        feats = self.model._get_dpt_embeddings(image_tensor)
        assert image_mask is not None, "Image mask must be provided for feature extraction."
        final_feats = []
        for i in range(len(image_mask)):
            if image_mask[i] == True:
                final_feats.append(feats[:, i, :, :])
        
        return torch.stack(final_feats, dim=1) if final_feats else feats


    def forward(self, image_tensor, image_mask):
        
        feats = self.extract_features(image_tensor, image_mask)
        return feats

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    model = DAV3Module().to("cuda")
    example_path = os.environ.get("DA3_EXAMPLE_IMAGE_DIR", "./assets/examples/SOH/")
    images = sorted(glob.glob(os.path.join(example_path, "*.png")))
    prediction = model.inference_images(images)

    depth = prediction.depth
    d = depth[0]
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(d_norm, cmap="magma")
    plt.colorbar(label="normalized depth")
    plt.title("Depth map")
    plt.axis("off")
    plt.show()
