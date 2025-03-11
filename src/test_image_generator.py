import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def generate_simple_image(size=(128, 128)):
    """Generate a very simple test image with basic shapes"""
    # Create blank white image
    image = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(image)

    # Draw a simple black rectangle
    rect_size = min(size[0], size[1]) // 2
    x1 = (size[0] - rect_size) // 2
    y1 = (size[1] - rect_size) // 2
    x2 = x1 + rect_size
    y2 = y1 + rect_size
    draw.rectangle([x1, y1, x2, y2], fill=0)

    return image


def generate_medium_image(size=(128, 128)):
    """Generate a medium complexity test image with gradient and circle"""
    image = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(image)

    # Background gradient
    for y in range(size[1]):
        for x in range(size[0]):
            # Simple linear gradient
            value = int(255 * (1 - y / size[1]))
            draw.point((x, y), fill=value)

    # Draw a circle
    radius = min(size) // 4
    center = (size[0] // 2, size[1] // 2)
    draw.ellipse([center[0] - radius, center[1] - radius,
                  center[0] + radius, center[1] + radius], fill=128)

    return image


def generate_complex_image(size=(128, 128)):
    """Generate a complex test image with multiple patterns"""
    image = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(image)

    # Background gradient
    for y in range(size[1]):
        for x in range(size[0]):
            value = int(255 * (1 - y / size[1]))
            draw.point((x, y), fill=value)

    # Concentric circles
    center = (size[0] // 2, size[1] // 2)
    for r in range(min(size) // 3, 0, -10):
        # Alternate between white and dark gray
        fill_color = 255 if (r // 10) % 2 == 0 else 64
        draw.ellipse([center[0] - r, center[1] - r,
                      center[0] + r, center[1] + r], fill=fill_color)

    # Grid pattern
    spacing = 16
    for x in range(0, size[0], spacing):
        draw.line([(x, 0), (x, size[1])], fill=128, width=1)
    for y in range(0, size[1], spacing):
        draw.line([(0, y), (size[0], y)], fill=128, width=1)

    # Text (if possible)
    try:
        font = ImageFont.load_default()
        draw.text((5, 5), "Test", font=font, fill=0)
    except:
        # Fallback if font not available
        draw.rectangle([5, 5, 25, 15], fill=0)

    # Apply slight blur for more natural appearance
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))

    return image


def generate_all_test_images():
    """Generate all test images and save them"""
    # Define sizes
    sizes = [(64, 64), (128, 128), (256, 256)]

    # Generate and save all test images
    for size in sizes:
        size_str = f"{size[0]}x{size[1]}"

        # Simple image
        simple = generate_simple_image(size)
        simple.save(f"test_images/simple_{size_str}.png")

        # Medium image
        medium = generate_medium_image(size)
        medium.save(f"test_images/medium_{size_str}.png")

        # Complex image
        complex_img = generate_complex_image(size)
        complex_img.save(f"test_images/complex_{size_str}.png")

    print("Generated all test images")
    return [
        "simple_64x64.png", "medium_64x64.png", "complex_64x64.png",
        "simple_128x128.png", "medium_128x128.png", "complex_128x128.png",
        "simple_256x256.png", "medium_256x256.png", "complex_256x256.png"
    ]


# When run directly, generate all images
if __name__ == "__main__":
    generate_all_test_images()