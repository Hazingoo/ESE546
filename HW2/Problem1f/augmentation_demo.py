import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def load_astronaut_image():
    """Load the astronaut image from scikit-image"""
    try:
        from skimage import data
        astronaut = data.astronaut()
        return Image.fromarray(astronaut)
    except ImportError:
        # Fallback: create a simple test image
        print("scikit-image not available, creating test image")
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(test_image)

def apply_augmentations(image):
    """Apply 10 different augmentations to the image"""
    
    # Define augmentations with reasonable parameters
    augmentations = {
        'Original': transforms.Compose([]),
        
        'ShearX': transforms.Compose([
            transforms.Lambda(lambda x: F.affine(x, angle=0, translate=[0, 0], scale=1.0, shear=[15, 0]))
        ]),
        
        'ShearY': transforms.Compose([
            transforms.Lambda(lambda x: F.affine(x, angle=0, translate=[0, 0], scale=1.0, shear=[0, 15]))
        ]),
        
        'TranslateX': transforms.Compose([
            transforms.Lambda(lambda x: F.affine(x, angle=0, translate=[20, 0], scale=1.0, shear=[0, 0]))
        ]),
        
        'TranslateY': transforms.Compose([
            transforms.Lambda(lambda x: F.affine(x, angle=0, translate=[0, 20], scale=1.0, shear=[0, 0]))
        ]),
        
        'Rotate': transforms.Compose([
            transforms.RandomRotation(degrees=30)
        ]),
        
        'Brightness': transforms.Compose([
            transforms.ColorJitter(brightness=0.5)
        ]),
        
        'Color': transforms.Compose([
            transforms.ColorJitter(saturation=0.5)
        ]),
        
        'Contrast': transforms.Compose([
            transforms.ColorJitter(contrast=0.5)
        ]),
        
        'Sharpness': transforms.Compose([
            transforms.Lambda(lambda x: F.adjust_sharpness(x, sharpness_factor=2.0))
        ]),
        
        'Posterize': transforms.Compose([
            transforms.Lambda(lambda x: F.posterize(x, bits=4))
        ]),
        
        'Solarize': transforms.Compose([
            transforms.Lambda(lambda x: F.solarize(x, threshold=128))
        ]),
        
        'Equalize': transforms.Compose([
            transforms.Lambda(lambda x: F.equalize(x))
        ])
    }
    
    results = {}
    for name, transform in augmentations.items():
        try:
            augmented = transform(image)
            results[name] = augmented
        except Exception as e:
            print(f"Error applying {name}: {e}")
            results[name] = image
    
    return results

def create_augmentation_grid(results, save_path='augmentation_results.png'):
    """Create a grid showing all augmentations"""
    names = list(results.keys())
    n_images = len(names)
    
    # Calculate grid dimensions
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, name in enumerate(names):
        row = i // cols
        col = i % cols
        
        ax = axes[row, col]
        ax.imshow(results[name])
        ax.set_title(f'{name}', fontsize=12)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Augmentation results saved to {save_path}")

def print_augmentation_parameters():
    """Print the parameters used for each augmentation"""
    print("\n=== Augmentation Parameters ===")
    print("ShearX: shear=[15, 0] degrees")
    print("ShearY: shear=[0, 15] degrees") 
    print("TranslateX: translate=[20, 0] pixels")
    print("TranslateY: translate=[0, 20] pixels")
    print("Rotate: degrees=30")
    print("Brightness: brightness=0.5")
    print("Color: saturation=0.5")
    print("Contrast: contrast=0.5")
    print("Sharpness: sharpness_factor=2.0")
    print("Posterize: bits=4")
    print("Solarize: threshold=128")
    print("Equalize: histogram equalization")

if __name__ == '__main__':
    print("Loading test image")
    image = load_astronaut_image()
    results = apply_augmentations(image)
    create_augmentation_grid(results)
    print_augmentation_parameters()
    
