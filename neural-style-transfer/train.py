import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from model import VGG
import os



imsize = 356
loader = transforms.Compose(
    [
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),

    ]
)
# Check if GPU is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to load and preprocess an image
def load_image(image_name, imsize=356):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

# Function to perform neural style transfer
def neural_style_transfer(original_path, style_path, output_path, total_steps=2000, alpha=1, beta=0.1):

    

    os.makedirs(output_path, exist_ok=True)
    # Load the original and style images
    original_img = load_image(original_path)
    style_img = load_image(style_path)

    # Initialize the canvas with the original image
    generated = original_img.clone().requires_grad_(True)

    # Load the pre-trained VGG model for feature extraction
    vgg_model = VGG().to(device).eval()

    # Hyperparameters for the optimization process
    learning_rate = 0.0001
    alpha = alpha 
    beta = beta
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
            generated_features, original_img_features, style_features):

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
            save_image(generated, f"{output_path}/generated_{step}.png")

        save_image(generated, f"{output_path}/generated_last.png")

    return f"{output_path}/generated_last.png"


