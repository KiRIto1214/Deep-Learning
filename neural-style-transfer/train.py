import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from model import VGG

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

# Check if GPU is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image size for processing
imsize = 356

# Transformation steps for preparing images
loader = transforms.Compose(
    [
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        # Uncomment the next line if you decide to use VGG normalization
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the original and style images
original_img = load_image(r"D:\Projects\Deep-Learning\neural-style-transfer\kid.jpeg")
style_img = load_image(r"D:\Projects\Deep-Learning\neural-style-transfer\style_to_transfer.jpg")

# Initialize the canvas with the original image
generated = original_img.clone().requires_grad_(True)

# Load the pre-trained VGG model for feature extraction
vgg_model = VGG().to(device).eval()

# Hyperparameters for the optimization process
total_steps = 2000
learning_rate = 0.0001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated], lr=learning_rate)

# Main optimization loop
for step in range(total_steps):
    # Extract features from the canvas, original, and style images
    generated_features = vgg_model(generated)
    original_img_features = vgg_model(original_img)
    style_features = vgg_model(style_img)

    # Initialize losses
    style_loss = original_loss = 0

    # Iterate through layers for style and originality
    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_img_features, style_features
    ):
        # Compute loss for preserving original content
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)

        # Compute loss for matching style
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )
        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )
        style_loss += torch.mean((G - A) ** 2)

    # Total loss as a combination of content and style losses
    total_loss = alpha * original_loss + beta * style_loss
    # Backpropagation and optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


    if step % 200 == 0:
        print(total_loss)
        save_image(generated, f"generated + {step}.png")

