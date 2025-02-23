import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


def generate_test_image(size=(256, 256)):
    """Generate a test image with patterns similar to reference"""
    image = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(image)

    # Draw complex patterns
    # Background gradient
    for y in range(size[1]):
        for x in range(size[0]):
            value = int(255 * (1 - y / size[1]))
            draw.point((x, y), fill=value)

    # Add some detailed patterns
    # Circles
    radius = min(size) // 4
    center = (size[0] // 2, size[1] // 2)
    for r in range(radius, 0, -20):
        draw.ellipse([center[0] - r, center[1] - r, center[0] + r, center[1] + r],
                     fill=int(255 * (r / radius)))

    # Grid pattern
    spacing = 20
    for x in range(0, size[0], spacing):
        draw.line([(x, 0), (x, size[1])], fill=128, width=1)
    for y in range(0, size[1], spacing):
        draw.line([(0, y), (size[0], y)], fill=128, width=1)

    # Add some text or symbols
    from PIL import ImageFont
    font = ImageFont.load_default()
    draw.text((10, 10), "Test Pattern", font=font, fill=0)

    # Apply some filtering for smoother appearance
    from PIL import ImageFilter
    image = image.filter(ImageFilter.GaussianBlur(radius=1))

    return image


def calculate_psnr(original, reconstructed):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def plot_radiation_pattern(angles, pattern, title="Radiation Pattern"):
    """Plot radiation pattern in polar coordinates"""
    angles_rad = np.deg2rad(angles)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(angles_rad, pattern)
    ax.set_title(title)
    ax.grid(True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    return fig


def calculate_ssim(original, reconstructed, window_size=11):
    """Calculate Structural Similarity Index (SSIM)"""
    from scipy.ndimage import uniform_filter

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    original = original.astype(float)
    reconstructed = reconstructed.astype(float)

    # Compute means
    mu1 = uniform_filter(original, window_size)
    mu2 = uniform_filter(reconstructed, window_size)

    # Compute variances and covariance
    sigma1_sq = uniform_filter(original ** 2, window_size) - mu1 ** 2
    sigma2_sq = uniform_filter(reconstructed ** 2, window_size) - mu2 ** 2
    sigma12 = uniform_filter(original * reconstructed, window_size) - mu1 * mu2

    # Compute SSIM
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(ssim_map)


def analyze_image_quality(original, reconstructed):
    """Comprehensive image quality analysis"""
    return {
        'psnr': calculate_psnr(original, reconstructed),
        'ssim': calculate_ssim(original, reconstructed),
        'mse': np.mean((original - reconstructed) ** 2),
        'mae': np.mean(np.abs(original - reconstructed))
    }


def plot_spectrum(signal, sampling_rate=1.0, title="Signal Spectrum"):
    """Plot frequency spectrum of signal"""
    frequencies = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    spectrum = np.fft.fft(signal)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies[:len(frequencies) // 2],
             20 * np.log10(np.abs(spectrum[:len(spectrum) // 2])))
    plt.grid(True)
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude (dB)')
    plt.title(title)
    return plt.gcf()


def plot_constellation_density(symbols, resolution=50, range_limit=2):
    """Plot constellation diagram with density information"""
    real_parts = np.real(symbols)
    imag_parts = np.imag(symbols)

    plt.figure(figsize=(10, 10))
    plt.hist2d(real_parts, imag_parts, bins=resolution,
               range=[[-range_limit, range_limit], [-range_limit, range_limit]])
    plt.colorbar(label='Density')
    plt.grid(True)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Constellation Density Plot')
    plt.axis('equal')
    return plt.gcf()


def save_transmission_results(original_image, received_image, results_dict,
                              output_dir='results'):
    """Save transmission results including images and metrics"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save images
    original_image.save(os.path.join(output_dir, 'original.png'))
    received_image.save(os.path.join(output_dir, 'received.png'))

    # Save metrics to text file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        for key, value in results_dict.items():
            f.write(f"{key}: {value}\n")


def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for transmission"""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')

    # Resize to target size
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)

    # Enhance contrast
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)

    # Apply slight sharpening
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

    return image