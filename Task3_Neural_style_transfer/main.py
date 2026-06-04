import os
import torch
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms

# 1. Simple Path Configurations
CONTENT_PATH = "my_photo.jpg"
STYLE_PATH = "art_style.jpg"
OUTPUT_PATH = "styled_output.jpg"
STEPS = 100  # Number of painting texture touch-up rounds

# 2. Image Processing Helpers
def load_image(path, img_size):
    """Loads an image file and shapes it into a mathematical tensor for the AI."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)

def save_image(tensor, output_path):
    """Converts the mathematical tensor back into a viewable picture file."""
    image = tensor.detach().cpu().squeeze(0).clamp(0, 1)
    to_pil = transforms.ToPILImage()
    to_pil(image).save(output_path)

# 3. The Texture Capture Formula (Gram Matrix)
def get_gram_matrix(feature_map):
    """Isolates raw artistic paint patterns while discarding object locations."""
    _, channels, height, width = feature_map.size()
    features = feature_map.view(channels, height * width)
    gram = torch.mm(features, features.t())
    return gram / (channels * height * width)

# 4. Main Neural Style Transfer Engine
def run_style_transfer():
    # Safety Check: Ensure the images actually exist in the folder
    if not os.path.exists(CONTENT_PATH) or not os.path.exists(STYLE_PATH):
        print(f"Error: Ensure '{CONTENT_PATH}' and '{STYLE_PATH}' are in your folder!")
        return

    # Automatically use GPU acceleration if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 200  # Resolution scale
    print(f"Using device: {device}")

    # Load baseline assets
    content_img = load_image(CONTENT_PATH, img_size).to(device)
    style_img = load_image(STYLE_PATH, img_size).to(device)
    
    # Target canvas begins as a clone of your photo, but allows pixel changes
    target_img = content_img.clone().requires_grad_(True).to(device)

    print("Loading VGG-19 AI Brain...")
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    
    # --- CRASH FIX: Disable in-place memory modifications on ReLU layers ---
    for layer in vgg.children():
        if isinstance(layer, torch.nn.ReLU):
            layer.inplace = False

    # Freeze the VGG brain parameters so we don't accidentally train the AI itself
    for param in vgg.parameters():
        param.requires_grad = False

    # Target layers to extract artistic details from VGG-19
    style_layers = ['0', '5', '10', '19', '28']  # Maps micro/macro paint strokes
    content_layer = '19'                          # Maps global object layout shapes

    # Pre-calculate reference traits for your photo and target style artwork
    content_features = {}
    style_grams = {}
    
    x_c, x_s = content_img, style_img
    for name, layer in vgg._modules.items():
        x_c, x_s = layer(x_c), layer(x_s)
        if name == content_layer:
            content_features[name] = x_c
        if name in style_layers:
            style_grams[name] = get_gram_matrix(x_s)

    # Set up the optimizer to tweak canvas pixel colors directly
    optimizer = optim.Adam([target_img], lr=0.03)

    print("Starting the blending process...")
    for step in range(1, STEPS + 1):
        x_t = target_img
        target_features = {}
        target_grams = {}
        
        # Pass the current morphing canvas snapshot through the VGG layers
        for name, layer in vgg._modules.items():
            x_t = layer(x_t)
            if name == content_layer:
                target_features[name] = x_t
            if name in style_layers:
                target_grams[name] = get_gram_matrix(x_t)

        # Calculate Loss Penalties
        content_loss = torch.mean((target_features[content_layer] - content_features[content_layer]) ** 2)
        
        style_loss = 0
        for layer_name in style_layers:
            style_loss += torch.mean((target_grams[layer_name] - style_grams[layer_name]) ** 2)

        # Balance weights: keep structural layout boundaries but emphasize textures
        total_loss = (1.0 * content_loss) + (1e5 * style_loss)

        # Backpropagation Routine: compute gradients and update image pixels
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Status Update Monitor
        if step % 50 == 0 or step == 1:
            print(f"Step {step}/{STEPS} | Content Loss: {content_loss.item():.4f} | Style Loss: {style_loss.item():.4f}")

    # Save output to your folder
    save_image(target_img, OUTPUT_PATH)
    print(f"\nSuccess! Open your folder to view '{OUTPUT_PATH}'")

if __name__ == "__main__":
    run_style_transfer()