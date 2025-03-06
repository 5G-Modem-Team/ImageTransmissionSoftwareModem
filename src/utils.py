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
    try:
        font = ImageFont.load_default()
        draw.text((10, 10), "Test Pattern", font=font, fill=0)
    except Exception:
        # Fallback if font is not available
        draw.rectangle([10, 10, 80, 25], outline=0, width=1)

    # Apply some filtering for smoother appearance
    image = image.filter(ImageFilter.GaussianBlur(radius=1))

    return image


def preprocess_image_for_transmission(image, target_size=(128, 128)):
    """
    Preprocess image to improve transmission reliability

    Args:
        image: Input PIL image
        target_size: Target size (smaller means fewer bits to transmit)

    Returns:
        Preprocessed PIL image
    """
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')

    # Resize to smaller dimensions to reduce data size
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)

    # Enhance contrast to improve robustness to noise
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)

    # Apply sharpening
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

    # Apply edge-preserving smoothing to reduce high-frequency components
    # that are more susceptible to noise
    image = image.filter(ImageFilter.SMOOTH)

    return image


def image_to_bit_stream(image, bit_depth=8):
    """
    Convert image to bit stream with error resilience

    Args:
        image: PIL Image
        bit_depth: Bit depth for representation (8 is standard, 4 for further reduction)

    Returns:
        tuple: (bit_stream, original_shape, metadata)
    """
    # Convert to NumPy array
    img_array = np.array(image)
    original_shape = img_array.shape

    if bit_depth < 8:
        # Reduce bit depth for smaller data size
        # First normalize to range 0-1
        img_normalized = img_array.astype(float) / 255.0

        # Quantize to fewer levels
        levels = 2 ** bit_depth
        img_quantized = np.floor(img_normalized * levels).astype(np.uint8)

        # Convert to bit_depth representation
        bit_stream = np.unpackbits(img_quantized)

        # Store metadata for reconstruction
        metadata = {
            'bit_depth': bit_depth,
            'levels': levels
        }
    else:
        # Standard 8-bit representation
        bit_stream = np.unpackbits(img_array.astype(np.uint8))
        metadata = {
            'bit_depth': 8,
            'levels': 256
        }

    # Ensure length is multiple of 6 (for 64QAM)
    if len(bit_stream) % 6 != 0:
        padding = 6 - (len(bit_stream) % 6)
        bit_stream = np.pad(bit_stream, (0, padding), 'constant')
        metadata['padding'] = padding
    else:
        metadata['padding'] = 0

    return bit_stream, original_shape, metadata


def bit_stream_to_image(bit_stream, original_shape, metadata):
    """
    Convert bit stream back to image with error handling

    Args:
        bit_stream: Received bit stream
        original_shape: Original image shape
        metadata: Metadata dictionary with bit_depth, levels, padding

    Returns:
        PIL Image
    """
    # Remove padding if any
    if 'padding' in metadata and metadata['padding'] > 0:
        bit_stream = bit_stream[:-metadata['padding']]

    # Calculate expected bits
    expected_bits = original_shape[0] * original_shape[1] * metadata.get('bit_depth', 8)

    # Handle bit stream length mismatch
    if len(bit_stream) < expected_bits:
        # Pad if too short
        bit_stream = np.pad(bit_stream, (0, expected_bits - len(bit_stream)), 'constant')
    elif len(bit_stream) > expected_bits:
        # Truncate if too long
        bit_stream = bit_stream[:expected_bits]

    # Reshape based on bit depth
    if metadata.get('bit_depth', 8) < 8:
        # Pack bits into uint8 array
        values = np.packbits(bit_stream)

        # Scale to full 8-bit range
        img_array = (values * (255 / (metadata['levels'] - 1))).astype(np.uint8)
    else:
        # Standard 8-bit representation
        values = np.packbits(bit_stream)
        img_array = values

    try:
        # Reshape to original dimensions
        img_array = img_array.reshape(original_shape)

        # Apply post-processing to reduce noise effects
        # Median filter to remove salt-and-pepper noise
        from scipy.ndimage import median_filter
        img_array = median_filter(img_array, size=2)
    except Exception as e:
        print(f"Error reshaping image: {e}")
        # Fallback: create blank image with original dimensions
        img_array = np.zeros(original_shape, dtype=np.uint8)

    return Image.fromarray(img_array)


def apply_error_resilience(bit_stream):
    """
    Apply simple error resilience techniques without full FEC

    Args:
        bit_stream: Original bit stream

    Returns:
        Modified bit stream with redundancy
    """
    # Simple bit repetition (repeat each bit 3 times)
    repeated = np.repeat(bit_stream, 3)

    # Add sync markers every 1000 bits
    marker = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)

    chunks = []
    for i in range(0, len(repeated), 1000):
        chunk = repeated[i:i + 1000]
        chunks.append(chunk)
        chunks.append(marker)

    # Combine all chunks
    result = np.concatenate(chunks)

    return result


def decode_with_error_correction(encoded_bits):
    """
    Decode bit stream with simple error correction

    Args:
        encoded_bits: Received bit stream with redundancy

    Returns:
        Corrected bit stream
    """
    # Remove sync markers (pattern: 10101010)
    marker = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    marker_len = len(marker)

    # Find sync marker positions
    i = 0
    data_chunks = []
    while i < len(encoded_bits):
        # Check if we found a marker
        if i <= len(encoded_bits) - marker_len and np.array_equal(encoded_bits[i:i + marker_len], marker):
            i += marker_len
        else:
            # Not a marker, add to data
            data_chunks.append(encoded_bits[i])
            i += 1

    # Reassemble data without markers
    data_without_markers = np.array(data_chunks)

    # Apply majority voting on each triplet of bits
    # Reshape to get triplets
    remainder = len(data_without_markers) % 3
    if remainder != 0:
        data_without_markers = data_without_markers[:-remainder]

    triplets = data_without_markers.reshape(-1, 3)

    # Majority vote
    corrected = np.sum(triplets, axis=1) >= 2

    return corrected.astype(np.uint8)


def calculate_psnr(original, reconstructed):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(original, reconstructed, window_size=11):
    """Calculate Structural Similarity Index (SSIM)"""
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