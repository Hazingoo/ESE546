import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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

def apply_augmentations(img):
    """Apply the 10 augmentations specified in the assignment (a-j)"""
    results = {}
    
    # Original image
    results['Original'] = img
    
    # (a) ShearX
    results['(a) ShearX'] = F.affine(img, angle=0, translate=[0,0], scale=1, shear=[20, 0])
    
    # (b) ShearY  
    results['(b) ShearY'] = F.affine(img, angle=0, translate=[0,0], scale=1, shear=[0, 20])
    
    # (c) TranslateX
    results['(c) TranslateX'] = F.affine(img, angle=0, translate=[30,0], scale=1, shear=[0, 0])
    
    # (d) TranslateY
    results['(d) TranslateY'] = F.affine(img, angle=0, translate=[0,30], scale=1, shear=[0, 0])
    
    # (e) Rotate
    results['(e) Rotate'] = F.rotate(img, angle=25)
    
    # (f) Brightness
    results['(f) Brightness'] = F.adjust_brightness(img, brightness_factor=1.5)
    
    # (g) Color (saturation)
    results['(g) Color'] = F.adjust_saturation(img, saturation_factor=3.0)
    
    # (h) Contrast
    results['(h) Contrast'] = F.adjust_contrast(img, contrast_factor=1.5)
    
    # (i) Sharpness
    results['(i) Sharpness'] = F.adjust_sharpness(img, sharpness_factor=2.0)
    
    # (j) Posterize
    results['(j) Posterize'] = F.posterize(img, bits=3)
    
    # Additional augmentations mentioned in assignment
    results['Solarize'] = F.solarize(img, threshold=100)
    results['Equalize'] = F.equalize(img)
    
    return results

def show_results(results):
    """Display all augmentation results in a grid"""
    names = list(results.keys())
    n_images = len(names)
    
    # Calculate grid size - use 4 columns for better fit
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif n_images == 1:
        axes = axes.reshape(1, 1)
    
    for i, name in enumerate(names):
        row = i // cols
        col = i % cols
        if rows == 1:
            ax = axes[col] if n_images > 1 else axes
        else:
            ax = axes[row, col]
        
        ax.imshow(results[name])
        ax.set_title(name, fontsize=10, pad=5)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        if rows == 1:
            axes[col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout(pad=1.0)
    plt.savefig('augmentations_complete.png', dpi=200, bbox_inches='tight')
    plt.show()

def print_parameters():
    """Print the parameters used for each augmentation"""
    print("\n=== Augmentation Parameters (Assignment a-j) ===")
    print("(a) ShearX: shear=[20, 0] degrees")
    print("(b) ShearY: shear=[0, 20] degrees") 
    print("(c) TranslateX: translate=[30, 0] pixels")
    print("(d) TranslateY: translate=[0, 30] pixels")
    print("(e) Rotate: angle=25 degrees")
    print("(f) Brightness: brightness_factor=1.5")
    print("(g) Color: saturation_factor=3.0")
    print("(h) Contrast: contrast_factor=1.5")
    print("(i) Sharpness: sharpness_factor=2.0")
    print("(j) Posterize: bits=3")
    print("\nAdditional augmentations:")
    print("Solarize: threshold=100")
    print("Equalize: histogram equalization")

if __name__ == '__main__':
    img = load_astronaut_image()
    results = apply_augmentations(img)
    show_results(results)
    print_parameters()
