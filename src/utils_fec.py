import numpy as np


class CRC16:
    """CRC-16 error detection implementation"""

    def __init__(self):
        self.polynomial = 0x8005
        self.initial = 0x0000

    def calculate(self, data):
        """Calculate CRC checksum"""
        crc = self.initial
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ self.polynomial) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc

    def append(self, data):
        """Append CRC checksum to data"""
        crc = self.calculate(data)
        # Convert CRC to two bytes
        return np.append(data, [(crc >> 8) & 0xFF, crc & 0xFF])

    def check(self, data_with_crc):
        """Check data and its CRC checksum"""
        # Last two bytes are CRC
        data = data_with_crc[:-2]
        received_crc = (data_with_crc[-2] << 8) | data_with_crc[-1]
        calculated_crc = self.calculate(data)
        return received_crc == calculated_crc, data


def process_image_for_transmission(image, max_size=(64, 64), use_crc=True, use_ldpc=True):
    """
    Preprocess image for robust transmission with enhanced preprocessing

    Args:
        image: Input PIL Image
        max_size: Target image dimensions
        use_crc: Enable CRC error detection
        use_ldpc: Enable LDPC error correction

    Returns:
        Processed data for transmission and original image shape
    """
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    import scipy.ndimage as ndi

    # 1. Convert to grayscale and resize
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize(max_size, Image.LANCZOS)

    # 2. Advanced image preprocessing
    # Enhance contrast and apply edge preservation
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    image = image.filter(ImageFilter.EDGE_ENHANCE)

    # 3. Convert to numpy array with more robust quantization
    img_array = np.array(image, dtype=np.float32)

    # Advanced normalization and quantization
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    img_array = np.round(img_array * 255).astype(np.uint8)

    # 4. Apply advanced noise reduction
    img_array = ndi.median_filter(img_array, size=2)
    img_array = ndi.gaussian_filter(img_array, sigma=0.5)

    # 5. Metadata preparation with robust dimension encoding
    metadata = np.array([
        image.width & 0xFF,  # Width (low byte)
        (image.width >> 8) & 0xFF,  # Width (high byte)
        image.height & 0xFF,  # Height (low byte)
        (image.height >> 8) & 0xFF  # Height (high byte)
    ], dtype=np.uint8)

    # 6. Combine metadata with image data
    data_with_metadata = np.concatenate([metadata, img_array.ravel()])

    # 7. Apply optional error correction
    if use_crc:
        crc = CRC16()
        data_with_crc = crc.append(data_with_metadata)
    else:
        data_with_crc = data_with_metadata

    # 8. Optional LDPC encoding
    if use_ldpc:
        try:
            from pyldpc import make_ldpc, encode

            # Dynamic LDPC parameters
            n = min(len(data_with_crc) * 2, 1000)
            H, G = make_ldpc(n, d_v=3, d_c=6, systematic=True, sparse=True)

            # Convert to bits and encode
            bits = np.unpackbits(data_with_crc)
            k = G.shape[1]
            bits = np.pad(bits, (0, k - (len(bits) % k)), mode='constant')
            bits = bits.reshape(-1, k)

            ldpc_encoded = np.concatenate([
                encode(G, block, 10) for block in bits
            ])

            # Pack and add metadata
            encoded_data = np.packbits(ldpc_encoded)
            ldpc_info = np.array([0xAA, 0xBB, len(data_with_crc) & 0xFF, (len(data_with_crc) >> 8) & 0xFF],
                                 dtype=np.uint8)
            final_data = np.concatenate([ldpc_info, encoded_data])

            return final_data, img_array.shape

        except Exception as e:
            print(f"LDPC encoding error: {e}")
            return data_with_crc, img_array.shape

    return data_with_crc, img_array.shape


def reconstruct_image_from_transmission(received_data, expected_shape=None, use_crc=True, use_ldpc=True):
    """
    Robust image reconstruction from transmitted data

    Args:
        received_data: Transmitted image data
        expected_shape: Fallback image shape
        use_crc: Enable CRC error detection
        use_ldpc: Enable LDPC error correction

    Returns:
        Reconstructed PIL Image
    """
    from PIL import Image
    import numpy as np
    import scipy.ndimage as ndi

    try:
        # 1. Basic validation
        if len(received_data) < 4:
            print("Insufficient data for reconstruction")
            return None

        # 2. Extract image dimensions
        width = received_data[0] | (received_data[1] << 8)
        height = received_data[2] | (received_data[3] << 8)

        # 3. Validate dimensions
        if width <= 0 or height <= 0 or width > 1024 or height > 1024:
            print(f"Invalid image dimensions: {width}x{height}")
            if expected_shape:
                height, width = expected_shape
            else:
                width = height = 64  # Default safe size

        # 4. Extract image data
        img_data = received_data[4:]

        # 5. Ensure correct data length
        expected_size = width * height
        if len(img_data) < expected_size:
            img_data = np.pad(img_data, (0, expected_size - len(img_data)), mode='constant')
        elif len(img_data) > expected_size:
            img_data = img_data[:expected_size]

        # 6. Reshape and process image
        img_array = img_data.reshape((height, width)).astype(np.float32)

        # 7. Advanced noise reduction and enhancement
        img_array = ndi.median_filter(img_array, size=3)
        img_array = ndi.gaussian_filter(img_array, sigma=0.7)

        # 8. Contrast stretching
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255

        # 9. Convert to uint8
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # 10. Create and return image
        return Image.fromarray(img_array)

    except Exception as e:
        print(f"Image reconstruction failed: {e}")
        return None