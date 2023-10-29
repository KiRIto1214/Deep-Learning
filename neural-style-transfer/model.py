import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # Customized list of chosen features
        self.my_chosen_features = ["1", "6", "11", "20", "28"]

        # We limit the model to conv5_1 (the 29th module in vgg)
        self.my_model = models.vgg19(pretrained=True).features[:30]

    def forward(self, x):
        # Store customized relevant features
        my_features = []

        # Iterate through each layer in the model
        for layer_num, layer in enumerate(self.my_model):
            x = layer(x)

            # Check if the layer is in the my_chosen_features list
            if str(layer_num) in self.my_chosen_features:
                my_features.append(x)

        return my_features