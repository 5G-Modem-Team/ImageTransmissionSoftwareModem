import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

from transmitter import Transmitter
from receiver import Receiver
from channel import Channel
from ofdm import OFDMProcessor, OFDMBeamformer
from ml_adaptive_mod import MLAdaptiveModulation
from adaptive_mod import AdaptiveModulation, ModulationType
from utils import generate_test_image, analyze_image_quality
from test_image_generator import generate_simple_image, generate_medium_image, generate_complex_image
from improved_ofdm import ImprovedOFDMProcessor  # Import the improved OFDM processor
from utils_fec import process_image_for_transmission, reconstruct_image_from_transmission  # Import FEC utilities


def test_ofdm(num_bits=10000, snr_db=20):
    """Test OFDM processing without beamforming"""
    print("\n=== Testing OFDM Processing ===")
    # Generate random bits for QPSK
    bits = np.random.randint(0, 2, num_bits - (num_bits % 2))

    # Initialize components
    tx = Transmitter(bits, modulation="QPSK")
    rx = Receiver(modulation="QPSK")
    ofdm = OFDMProcessor()

    # Modulate to symbols
    tx_symbols = tx.transmit(beam_angle=None)

    # Apply OFDM
    print("Applying OFDM modulation...")
    ofdm_signal = ofdm.modulate(tx_symbols)

    # Add AWGN noise
    signal_power = np.mean(np.abs(ofdm_signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*ofdm_signal.shape) +
                                        1j * np.random.randn(*ofdm_signal.shape))
    rx_ofdm = ofdm_signal + noise

    # Demodulate OFDM
    print("Applying OFDM demodulation...")
    rx_symbols = ofdm.demodulate(rx_ofdm)

    # Account for potential length differences
    min_len = min(len(tx_symbols), len(rx_symbols))
    tx_symbols_trunc = tx_symbols[:min_len]
    rx_symbols_trunc = rx_symbols[:min_len]

    # Calculate symbol error
    symbol_error = np.mean(np.abs(tx_symbols_trunc - rx_symbols_trunc) ** 2)
    print(f"Average Symbol Error: {symbol_error:.6f}")

    # Demodulate and calculate BER
    rx_bits, _ = rx.receive(rx_symbols)
    min_len = min(len(bits), len(rx_bits))
    ber = np.mean(bits[:min_len] != rx_bits[:min_len])
    print(f"OFDM BER: {ber:.6f}")

    # Plot original vs received symbols
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.scatter(np.real(tx_symbols[:100]), np.imag(tx_symbols[:100]), alpha=0.7)
    plt.grid(True)
    plt.title("Original Symbols")
    plt.axis('equal')

    plt.subplot(122)
    plt.scatter(np.real(rx_symbols[:100]), np.imag(rx_symbols[:100]), alpha=0.7)
    plt.grid(True)
    plt.title("Received Symbols after OFDM")
    plt.axis('equal')

    plt.savefig("results/ofdm_test.png")
    plt.close()

    return ber


def test_beamforming_channel(num_bits=10000, snr_db=20, beam_angle=30):
    """Test beamforming and channel effects"""
    print("\n=== Testing Beamforming and Channel ===")
    # Generate random bits for QPSK
    bits = np.random.randint(0, 2, num_bits - (num_bits % 2))

    # Initialize components
    tx = Transmitter(bits, modulation="QPSK", num_antennas=4)
    rx = Receiver(modulation="QPSK", num_antennas=4)
    channel = Channel(num_tx_antennas=4, num_rx_antennas=4)

    # Modulate and apply beamforming
    print(f"Applying beamforming at {beam_angle} degrees...")
    tx_symbols = tx.transmit(beam_angle=beam_angle)

    # Plot radiation pattern
    plt.figure(figsize=(10, 6))
    tx.plot_radiation_pattern(tx_symbols)
    plt.savefig("results/beamforming_pattern.png")
    plt.close()

    # Apply channel effects
    print("Applying channel effects...")
    rx_symbols = channel.apply_channel(tx_symbols, snr_db)

    # Add debug info
    print(f"tx_symbols shape: {tx_symbols.shape}")
    print(f"rx_symbols shape: {rx_symbols.shape}")
    print(f"channel matrix shape: {channel.get_channel_matrix().shape}")

    try:
        # Equalize and demodulate
        rx_bits, eq_symbols = rx.receive(rx_symbols, channel.get_channel_matrix())

        # Calculate BER
        min_len = min(len(bits), len(rx_bits))
        ber = np.mean(bits[:min_len] != rx_bits[:min_len])
        print(f"Beamforming + Channel BER: {ber:.6f}")

        # Plot received constellation
        plt.figure(figsize=(8, 8))
        # Ensure we're plotting a reasonable number of points
        plot_symbols = eq_symbols.flatten()[:100]
        plt.scatter(np.real(plot_symbols), np.imag(plot_symbols), alpha=0.7)
        plt.grid(True)
        plt.title(f"Received Constellation after Beamforming and Channel")
        plt.axis('equal')
        plt.savefig("results/beamforming_channel_constellation.png")
        plt.close()

        return ber

    except Exception as e:
        print(f"Error during receive processing: {str(e)}")
        print("Continuing with other tests...")
        return 1.0  # Return a high BER to indicate failure


def test_adaptive_modulation(snr_range=[5, 10, 15, 20, 25, 30]):
    """Test adaptive modulation selection"""
    print("\n=== Testing Adaptive Modulation ===")
    adaptive_mod = MLAdaptiveModulation()  # Use ML version

    # Generate visualization of ML training results
    ml_vis_file = adaptive_mod.visualize_training_results()
    print(f"ML training visualization saved to: {ml_vis_file}")

    results = []
    for snr in snr_range:
        # Create test channel matrix
        test_channel = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        noise_variance = 10 ** (-snr / 10)

        # Test with different data sizes
        small_data = adaptive_mod.adapt_to_channel(test_channel, noise_variance, data_size=1000)
        medium_data = adaptive_mod.adapt_to_channel(test_channel, noise_variance, data_size=50000)
        large_data = adaptive_mod.adapt_to_channel(test_channel, noise_variance, data_size=200000)

        # Use medium data result for plotting
        mod_type = medium_data['base_modulation']
        capacity = medium_data['channel_capacity']
        bits_per_sym = medium_data['actual_rate']

        print(
            f"SNR: {snr}dB -> Selected: {mod_type.name}, Bits/Symbol: {bits_per_sym}, Capacity: {capacity:.2f} bits/s/Hz")
        print(f"  Small data ({1000} bits): {small_data['base_modulation'].name}")
        print(f"  Medium data ({50000} bits): {medium_data['base_modulation'].name}")
        print(f"  Large data ({200000} bits): {large_data['base_modulation'].name}")

        results.append((snr, mod_type.name, bits_per_sym, capacity))

    # Plot adaptive modulation selection
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    snrs = [r[0] for r in results]
    bits_per_sym = [r[2] for r in results]
    plt.step(snrs, bits_per_sym, where='post', linewidth=2)
    plt.ylabel("Bits per Symbol")
    plt.grid(True)
    plt.title("Adaptive Modulation Selection")

    plt.subplot(212)
    capacities = [r[3] for r in results]
    plt.plot(snrs, capacities, 'o-', linewidth=2)
    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Channel Capacity (bits/s/Hz)")

    plt.tight_layout()
    plt.savefig("results/adaptive_modulation.png")
    plt.close()

    return results


def test_adaptive_comparison(image_size=(64, 64), snr_range=[10, 15, 20, 25, 30]):
    """Compare ML adaptive vs traditional adaptive under challenging channel conditions"""
    print("\n=== Comparing Adaptive Modulation Techniques ===")

    # Results storage
    ml_results = []
    trad_results = []

    # Create test image
    test_image = generate_test_image(size=image_size)

    for snr_db in snr_range:
        print(f"\n--- Testing at SNR {snr_db}dB ---")

        # Test with traditional adaptive modulation
        print("Running traditional adaptive modulation...")
        trad_ber, trad_quality, _ = test_full_system(
            input_image=test_image,
            image_size=image_size,
            modulation="traditional",
            use_ofdm=True,
            use_ldpc=False,
            use_crc=True,
            snr_db=snr_db
        )

        # Test with ML adaptive modulation
        print("Running ML adaptive modulation...")
        ml_ber, ml_quality, _ = test_full_system(
            input_image=test_image,
            image_size=image_size,
            modulation="ml_adaptive",
            use_ofdm=True,
            use_ldpc=False,
            use_crc=True,
            snr_db=snr_db
        )

        # Record results
        trad_results.append({
            'snr': snr_db,
            'ber': trad_ber,
            'psnr': trad_quality['psnr'],
            'ssim': trad_quality['ssim']
        })

        ml_results.append({
            'snr': snr_db,
            'ber': ml_ber,
            'psnr': ml_quality['psnr'],
            'ssim': ml_quality['ssim']
        })

    # Create comparison plots
    plt.figure(figsize=(15, 10))

    # BER comparison
    plt.subplot(2, 1, 1)
    plt.semilogy([r['snr'] for r in trad_results], [r['ber'] for r in trad_results], 'o-', label='Traditional')
    plt.semilogy([r['snr'] for r in ml_results], [r['ber'] for r in ml_results], 's-', label='ML-based')
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER Comparison: ML vs Traditional Adaptive Modulation')
    plt.legend()

    # PSNR comparison
    plt.subplot(2, 1, 2)
    plt.plot([r['snr'] for r in trad_results], [r['psnr'] for r in trad_results], 'o-', label='Traditional')
    plt.plot([r['snr'] for r in ml_results], [r['psnr'] for r in ml_results], 's-', label='ML-based')
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.title('Image Quality Comparison: ML vs Traditional Adaptive Modulation')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/adaptive_comparison.png')

    # Print summary table
    print("\n=== Adaptive Modulation Comparison ===")
    print(f"{'SNR':<6} {'Trad BER':<12} {'ML BER':<12} {'Trad PSNR':<12} {'ML PSNR':<12}")
    print("-" * 60)

    for i in range(len(snr_range)):
        print(f"{snr_range[i]:<6} {trad_results[i]['ber']:<12.6f} {ml_results[i]['ber']:<12.6f} "
              f"{trad_results[i]['psnr']:<12.2f} {ml_results[i]['psnr']:<12.2f}")

    return trad_results, ml_results


def test_full_system(input_image=None, image_size=(128, 128), modulation="ml_adaptive",
                     use_ofdm=True, use_adaptive=True, use_ldpc=True, use_crc=True,
                     beam_angle=30, snr_db=30):
    """Test the full image transmission system with improved reliability"""
    print("\n=== Testing Full System ===")

    # Generate or load a test image
    print("Preparing test image...")
    if input_image is None:
        if image_size != (128, 128):
            image = generate_test_image(size=image_size)
        else:
            image = generate_test_image()
        image_description = f"default_{image_size[0]}x{image_size[1]}"
    elif isinstance(input_image, str):
        if input_image == "simple":
            image = generate_simple_image(size=image_size)
            image_description = f"simple_{image_size[0]}x{image_size[1]}"
        elif input_image == "medium":
            image = generate_medium_image(size=image_size)
            image_description = f"medium_{image_size[0]}x{image_size[1]}"
        elif input_image == "complex":
            image = generate_complex_image(size=image_size)
            image_description = f"complex_{image_size[0]}x{image_size[1]}"
        else:
            try:
                image = Image.open(input_image).convert('L')
                image = image.resize(image_size, Image.LANCZOS)  # Resize to target size
                image_description = os.path.basename(input_image)
            except:
                image = generate_test_image(size=image_size)
                image_description = "default (fallback)"
    else:
        image = input_image
        image_description = "provided"

    # Save original image
    image.save(f"original_{image_description}.png")

    # Default fallback for bit stream conversion
    orig_shape = None

    # Process image for transmission with improved preprocessing
    try:
        if use_crc or use_ldpc:
            # Use enhanced preprocessing with error detection/correction
            from utils_fec import process_image_for_transmission, reconstruct_image_from_transmission
            bit_stream, orig_shape = process_image_for_transmission(
                image, max_size=image_size, use_crc=use_crc, use_ldpc=use_ldpc
            )

            # Store the original image dimensions for verification
            width, height = image.size
            print(f"Original image dimensions: {width}x{height}, shape: {orig_shape}")

            # Safely convert to bits
            if not isinstance(bit_stream, np.ndarray):
                bit_stream = np.frombuffer(bit_stream, dtype=np.uint8)

            bit_stream = np.unpackbits(bit_stream.astype(np.uint8))
        else:
            # Standard conversion to bits
            img_array = np.array(image)
            orig_shape = img_array.shape
            print(f"Original image array shape: {orig_shape}")
            bit_stream = np.unpackbits(img_array.astype(np.uint8))
    except Exception as e:
        print(f"Error in bit stream conversion: {e}")
        # Fallback to standard conversion
        img_array = np.array(image)
        orig_shape = img_array.shape
        print(f"Fallback to original image array shape: {orig_shape}")
        bit_stream = np.unpackbits(img_array.astype(np.uint8))

    # Ensure bit_stream is defined and valid
    if not isinstance(bit_stream, np.ndarray) or len(bit_stream) == 0:
        # Absolute fallback if everything else fails
        img_array = np.array(image)
        bit_stream = np.unpackbits(img_array.astype(np.uint8))
        orig_shape = img_array.shape

    # Initialize channel - use a more challenging channel type
    channel = Channel(num_tx_antennas=4, num_rx_antennas=4, channel_type='frequency_selective')

    # Select modulation - use ML-based selection or force specific modulation
    if modulation == "ml_adaptive":
        adaptive_mod = MLAdaptiveModulation()  # ML version
        channel_state = {
            'H': channel.get_channel_matrix(),
            'noise_variance': 10 ** (-snr_db / 10)
        }

        # Standard adaptation
        adaptation = adaptive_mod.adapt_to_channel(
            channel_state['H'],
            channel_state['noise_variance'],
            data_size=len(bit_stream)
        )

        selected_mod = adaptation['base_modulation'].name
        estimated_ber = adaptation.get('estimated_ber', 0.0)
        print(f"ML selected modulation: {selected_mod}, Estimated BER: {estimated_ber:.6f}")
        modulation = selected_mod

    elif modulation == "traditional":
        traditional_mod = AdaptiveModulation()  # Traditional rule-based version
        channel_state = {
            'H': channel.get_channel_matrix(),
            'noise_variance': 10 ** (-snr_db / 10)
        }

        # Traditional adaptation
        adaptation = traditional_mod.adapt_to_channel(
            channel_state['H'],
            channel_state['noise_variance'],
            data_size=len(bit_stream)
        )

        selected_mod = adaptation['base_modulation'].name
        print(f"Traditional adaptive selected modulation: {selected_mod}")
        modulation = selected_mod

    elif modulation == "auto" and use_adaptive:
        adaptive_mod = MLAdaptiveModulation()  # Use ML version
        channel_state = {
            'H': channel.get_channel_matrix(),
            'noise_variance': 10 ** (-snr_db / 10)
        }

        # Pass data_size (bit stream length) to adapt_to_channel
        if use_ldpc:
            # With LDPC, use the FEC-aware adaptation
            adaptation = adaptive_mod.adapt_with_fec(
                channel_state['H'],
                channel_state['noise_variance'],
                data_size=len(bit_stream),
                enable_ldpc=use_ldpc
            )
        else:
            # Standard adaptation
            adaptation = adaptive_mod.adapt_to_channel(
                channel_state['H'],
                channel_state['noise_variance'],
                data_size=len(bit_stream)
            )

        selected_mod = adaptation['base_modulation'].name
        estimated_ber = adaptation.get('estimated_ber', 0.0)
        print(f"ML selected modulation: {selected_mod}, Estimated BER: {estimated_ber:.6f}")
        modulation = selected_mod  # Use the ML-selected modulation
    elif modulation == "auto":
        # If auto but ML not enabled, default to BPSK for image data
        modulation = "BPSK"
        print(f"Auto-selecting modulation (no ML): {modulation}")

    # Create transmitter and receiver
    tx = Transmitter(bit_stream, modulation=modulation, num_antennas=4)
    rx = Receiver(modulation=modulation, num_antennas=4)

    # Modulate with beamforming
    print(f"Modulating with {modulation} and beamforming...")
    tx_symbols = tx.transmit(beam_angle=beam_angle)

    # Apply OFDM if enabled
    if use_ofdm:
        print("Applying OFDM with improved synchronization...")
        # Use improved OFDM processor
        ofdm = ImprovedOFDMProcessor()
        ofdm_beamformer = OFDMBeamformer(4)

        # Process each antenna separately for OFDM
        if len(tx_symbols.shape) > 1:
            # For beamforming case (time_samples, antennas)
            ofdm_signals = []
            for ant in range(tx_symbols.shape[1]):
                ofdm_signals.append(ofdm.modulate(tx_symbols[:, ant]))

            # Apply beamforming to OFDM signals
            max_len = max(len(sig) for sig in ofdm_signals)
            # Pad signals to same length
            padded_signals = []
            for sig in ofdm_signals:
                padded_signals.append(np.pad(sig, (0, max_len - len(sig)), 'constant'))

            # Stack signals and apply OFDM beamforming
            stacked_signal = np.stack(padded_signals, axis=1)
            tx_symbols = stacked_signal
        else:
            # Single antenna case
            tx_symbols = ofdm.modulate(tx_symbols)
            # Apply beamforming
            tx_symbols = ofdm_beamformer.apply_beamforming(tx_symbols, beam_angle)

    # Apply channel effects
    print(f"Applying channel effects with SNR: {snr_db}dB...")
    rx_symbols = channel.apply_channel(tx_symbols, snr_db)

    # Demodulate OFDM if used
    if use_ofdm:
        print("Demodulating OFDM with improved synchronization...")
        if len(rx_symbols.shape) > 1:
            # For multi-antenna case
            rx_signals = []
            # Process each antenna separately
            for ant in range(rx_symbols.shape[1]):
                rx_signals.append(ofdm.demodulate(rx_symbols[:, ant]))

            # Stack signals
            max_len = max(len(sig) for sig in rx_signals)
            # Pad signals to same length
            padded_signals = []
            for sig in rx_signals:
                padded_signals.append(np.pad(sig, (0, max_len - len(sig)), 'constant'))

            # Stack signals
            rx_symbols = np.stack(padded_signals, axis=1)
        else:
            # Single antenna case
            rx_symbols = ofdm.demodulate(rx_symbols)

    # Print debug info
    print(f"Received symbols shape before equalization: {rx_symbols.shape}")

    # Receive and demodulate
    print("Equalizing and demodulating...")
    rx_bits, _ = rx.receive(rx_symbols, channel.get_channel_matrix())

    # Calculate BER on the bits
    min_len = min(len(bit_stream), len(rx_bits))
    ber = np.mean(bit_stream[:min_len] != rx_bits[:min_len])
    print(f"Full System BER: {ber:.6f}")

    # Convert bits back to image
    print("Converting bits back to image...")

    # Convert bits back to image
    print("Converting bits back to image...")

    if use_crc or use_ldpc:
        # Use enhanced reconstruction
        from utils_fec import reconstruct_image_from_transmission
        # Convert bits to bytes
        rx_bytes = np.packbits(rx_bits)
        print(f"Received data size: {len(rx_bytes)} bytes")

        rx_image = reconstruct_image_from_transmission(
            rx_bytes, expected_shape=orig_shape, use_crc=use_crc, use_ldpc=use_ldpc
        )

        if rx_image is None:
            # Fallback if reconstruction fails
            print("FEC-based image reconstruction failed, using standard method")
            # Standard reconstruction
            expected_bits = orig_shape[0] * orig_shape[1] * 8
            if len(rx_bits) < expected_bits:
                rx_bits = np.pad(rx_bits, (0, expected_bits - len(rx_bits)), 'constant')
            elif len(rx_bits) > expected_bits:
                rx_bits = rx_bits[:expected_bits]

            rx_bytes = np.packbits(rx_bits)
            try:
                rx_img_array = rx_bytes.reshape(orig_shape)
                from scipy.ndimage import median_filter
                rx_img_array = median_filter(rx_img_array, size=2)
                rx_image = Image.fromarray(rx_img_array)
            except Exception as e:
                print(f"Standard reconstruction also failed: {e}")
                # Create a blank image as absolute last resort
                rx_image = Image.new('L', (64, 64), 128)
    else:
        # Standard bit-to-image conversion
        expected_bits = orig_shape[0] * orig_shape[1] * 8
        print(f"Required bits: {expected_bits}, Received bits: {len(rx_bits)}")

        if len(rx_bits) < expected_bits:
            # Pad if too short
            print(f"Padding {expected_bits - len(rx_bits)} bits")
            rx_bits = np.pad(rx_bits, (0, expected_bits - len(rx_bits)), 'constant')
        elif len(rx_bits) > expected_bits:
            # Truncate if too long
            print(f"Truncating {len(rx_bits) - expected_bits} bits")
            rx_bits = rx_bits[:expected_bits]

        # Convert to bytes and reshape
        rx_bytes = np.packbits(rx_bits)
        try:
            rx_img_array = rx_bytes.reshape(orig_shape)
            # Apply median filter to reduce salt-and-pepper noise
            from scipy.ndimage import median_filter
            rx_img_array = median_filter(rx_img_array, size=2)
        except Exception as e:
            print(f"Error reshaping image: {e}")
            # Create blank image as fallback
            rx_img_array = np.zeros(orig_shape, dtype=np.uint8)

        rx_image = Image.fromarray(rx_img_array)

    # Save received image
    rx_image.save(f"results/received_{image_description}.png")

    # Calculate image quality metrics
    quality = analyze_image_quality(np.array(image), np.array(rx_image))
    print("Image Quality Metrics:")
    for metric, value in quality.items():
        print(f"{metric}: {value:.2f}")

    # Create a comparison image
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(np.array(image), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(np.array(rx_image), cmap='gray')
    plt.title(f"Received Image\nBER: {ber:.6f}, PSNR: {quality['psnr']:.2f}dB")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"results/comparison_{image_description}.png")
    plt.close()

    return ber, quality, image_description


def visualize_ldpc_performance():
    """Create visualization of LDPC performance improvement"""
    import matplotlib.pyplot as plt
    import numpy as np
    from pyldpc import make_ldpc, encode, decode, get_message

    # Generate random message
    n = 200  # Codeword length
    d_v = 2  # Variable node degree
    d_c = 5  # Check node degree

    # Create LDPC code
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]  # Message length

    # Random message
    v = np.random.randint(0, 2, k)

    # Test over different SNR values
    snr_range = np.linspace(0, 10, 11)
    ber_with_ldpc = []
    ber_without_ldpc = []

    for snr in snr_range:
        # Encode using LDPC
        y = encode(G, v, snr)  # Encoded with noise
        d = decode(H, y, snr)  # Decoded
        x = get_message(G, d)  # Extract message

        # Calculate BER with LDPC
        bit_errors_ldpc = np.sum(v != x) / k
        ber_with_ldpc.append(bit_errors_ldpc)

        # Direct transmission (no coding)
        # Simulate BPSK modulation with noise
        direct_signal = 2 * v - 1  # BPSK modulation
        noise_power = 10 ** (-snr / 10)
        noise = np.sqrt(noise_power) * np.random.randn(len(v))
        received_signal = direct_signal + noise
        received_bits = (received_signal > 0).astype(int)

        # Calculate BER without LDPC
        bit_errors_nocoding = np.sum(v != received_bits) / k
        ber_without_ldpc.append(bit_errors_nocoding)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_without_ldpc, 'o-', label='Without LDPC')
    plt.semilogy(snr_range, ber_with_ldpc, 's-', label='With LDPC')
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('LDPC Coding Performance Comparison')
    plt.legend()
    plt.savefig("results/ldpc_performance.png")
    plt.close()

    print("LDPC performance comparison saved to results/ldpc_performance.png")

    return "results/ldpc_performance.png"

def test_multiple_images(snr_db=25):
    """Test the system with multiple images of different complexity"""
    print("\n=== Testing Multiple Images ===")

    # Define test images
    test_images = ["simple", "medium", "complex"]

    # Results table
    results = []

    for img_type in test_images:
        print(f"\n--- Testing {img_type} image ---")
        ber, quality, desc = test_full_system(
            input_image=img_type,
            use_ofdm=True,
            use_adaptive=True,
            snr_db=snr_db
        )
        results.append({
            'type': img_type,
            'ber': ber,
            'psnr': quality['psnr'],
            'ssim': quality['ssim']
        })

    # Print results table
    print("\n=== Results Summary ===")
    print(f"{'Image Type':<15} {'BER':<10} {'PSNR (dB)':<15} {'SSIM':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['type']:<15} {r['ber']:<10.6f} {r['psnr']:<15.2f} {r['ssim']:<10.4f}")

    return results


def test_simple_image_transmission():
    """Test image transmission with minimal complexity to isolate issues"""
    # Import required libraries
    from PIL import Image, ImageDraw
    import numpy as np

    # Create a very simple test image - just black and white squares
    image = Image.new('L', (64, 64), 255)  # White background
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 54, 54], fill=0)  # Black square in the center

    # Save original
    image.save("results/test_simple_original.png")

    # Convert to bits directly, without any preprocessing
    img_array = np.array(image)
    original_shape = img_array.shape
    bit_stream = np.unpackbits(img_array.astype(np.uint8))

    # Use BPSK without OFDM or beamforming
    tx = Transmitter(bit_stream, modulation="BPSK", num_antennas=1)
    rx = Receiver(modulation="BPSK", num_antennas=1)

    # Simple transmission - no OFDM
    tx_symbols = tx.transmit(beam_angle=None)

    # Simple AWGN channel - high SNR for testing
    channel = Channel(num_tx_antennas=1, num_rx_antennas=1, channel_type='awgn')
    rx_symbols = channel.add_awgn(tx_symbols, 30)  # High SNR = 30dB

    # Demodulate
    rx_bits, _ = rx.receive(rx_symbols)

    # Calculate BER
    min_len = min(len(bit_stream), len(rx_bits))
    ber = np.mean(bit_stream[:min_len] != rx_bits[:min_len])
    print(f"Test Simple - BER: {ber:.6f}")

    # Ensure bit lengths match
    if len(rx_bits) < len(bit_stream):
        rx_bits = np.pad(rx_bits, (0, len(bit_stream) - len(rx_bits)), 'constant')
    else:
        rx_bits = rx_bits[:len(bit_stream)]

    # Convert back to image
    rx_bytes = np.packbits(rx_bits)
    rx_img_array = rx_bytes.reshape(original_shape)
    rx_image = Image.fromarray(rx_img_array)
    rx_image.save("results/test_simple_received.png")

    return ber


def test_direct_image_transmission(image_size=(64, 64), snr_db=30):
    """Test the entire image transmission process with direct control"""
    # Import necessary libraries
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate a simple test pattern
    image = generate_test_image(size=image_size)

    # Save original image
    image.save("results/direct_test_original.png")
    print(f"Original image size: {image.size}")

    # 1. DIRECTLY convert image to numpy array
    img_array = np.array(image)
    original_shape = img_array.shape
    print(f"Original array shape: {original_shape}")

    # 2. DIRECTLY convert to bit stream
    bit_stream = np.unpackbits(img_array.astype(np.uint8))
    print(f"Bit stream length: {len(bit_stream)}")

    # 3. Use BPSK modulation for simplicity
    tx = Transmitter(bit_stream, modulation="BPSK", num_antennas=1)
    rx = Receiver(modulation="BPSK", num_antennas=1)

    # 4. Simple transmission without OFDM or beamforming
    tx_symbols = tx.transmit(beam_angle=None)

    # 5. Simple AWGN channel with high SNR for testing
    channel = Channel(num_tx_antennas=1, num_rx_antennas=1, channel_type='awgn')
    rx_symbols = channel.add_awgn(tx_symbols, snr_db)

    # 6. Demodulate
    rx_bits, _ = rx.receive(rx_symbols)
    print(f"Received bits length: {len(rx_bits)}")

    # 7. Calculate BER
    min_len = min(len(bit_stream), len(rx_bits))
    ber = np.mean(bit_stream[:min_len] != rx_bits[:min_len])
    print(f"BER: {ber:.6f}")

    # 8. ENSURE exact bit length match
    if len(rx_bits) < len(bit_stream):
        rx_bits = np.pad(rx_bits, (0, len(bit_stream) - len(rx_bits)), 'constant')
    else:
        rx_bits = rx_bits[:len(bit_stream)]

    # 9. DIRECTLY convert bits back to image
    rx_bytes = np.packbits(rx_bits)
    print(f"Packed bytes length: {len(rx_bytes)}, Expected shape product: {original_shape[0] * original_shape[1]}")

    if len(rx_bytes) != original_shape[0] * original_shape[1]:
        print("WARNING: Byte count mismatch with image dimensions")

    # Handle mismatch
    if len(rx_bytes) < original_shape[0] * original_shape[1]:
        rx_bytes = np.pad(rx_bytes, (0, original_shape[0] * original_shape[1] - len(rx_bytes)), 'constant')
    else:
        rx_bytes = rx_bytes[:original_shape[0] * original_shape[1]]

    # 10. Reshape and create image
    rx_img_array = rx_bytes.reshape(original_shape)
    rx_image = Image.fromarray(rx_img_array)
    rx_image.save("results/direct_test_received.png")

    # 11. Calculate image quality metrics
    quality = analyze_image_quality(img_array, rx_img_array)

    # 12. Create comparison image
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(img_array, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(rx_img_array, cmap='gray')
    plt.title(f"Received Image\nBER: {ber:.6f}, PSNR: {quality['psnr']:.2f}dB")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("results/direct_test_comparison.png")
    plt.close()

    return ber, quality


def debug_ofdm_transmission(image_size=(32, 32), snr_db=30):
    """Debug OFDM-specific issues by breaking down the process"""
    # Import required libraries
    from PIL import Image, ImageDraw
    import numpy as np

    # Generate a very simple image
    image = Image.new('L', image_size, 255)  # White background
    draw = ImageDraw.Draw(image)
    draw.rectangle([5, 5, image_size[0] - 5, image_size[1] - 5], fill=0)  # Black frame

    # Save original
    image.save("results/debug_ofdm_original.png")

    # Convert to bits
    img_array = np.array(image)
    original_shape = img_array.shape
    bit_stream = np.unpackbits(img_array.astype(np.uint8))

    # Print diagnostic info
    print(f"Original image dimensions: {image_size}")
    print(f"Array shape: {original_shape}")
    print(f"Bit stream length: {len(bit_stream)}")

    # Create transmitter with BPSK
    tx = Transmitter(bit_stream, modulation="BPSK", num_antennas=1)
    rx = Receiver(modulation="BPSK", num_antennas=1)

    # Generate symbols
    tx_symbols = tx.transmit(beam_angle=None)
    print(f"Transmitted symbols shape: {tx_symbols.shape}")

    # Create OFDM processor
    ofdm = ImprovedOFDMProcessor()

    # OFDM modulation
    ofdm_signal = ofdm.modulate(tx_symbols)
    print(f"OFDM signal length: {len(ofdm_signal)}")

    # Simple AWGN channel
    channel = Channel(num_tx_antennas=1, num_rx_antennas=1, channel_type='awgn')
    rx_ofdm = channel.add_awgn(ofdm_signal, snr_db)

    # OFDM demodulation
    rx_symbols = ofdm.demodulate(rx_ofdm)
    print(f"Received symbols length: {len(rx_symbols)}")

    # Demodulate symbols to bits
    rx_bits, _ = rx.receive(rx_symbols)
    print(f"Received bits length: {len(rx_bits)}")

    # Calculate BER
    min_len = min(len(bit_stream), len(rx_bits))
    ber = np.mean(bit_stream[:min_len] != rx_bits[:min_len])
    print(f"BER: {ber:.6f}")
    print(f"Bit length ratio (rx/tx): {len(rx_bits) / len(bit_stream):.4f}")

    # Check if we need padding
    if len(rx_bits) < len(bit_stream):
        missing = len(bit_stream) - len(rx_bits)
        percent_missing = (missing / len(bit_stream)) * 100
        print(f"Missing {missing} bits ({percent_missing:.2f}% of original)")
        rx_bits = np.pad(rx_bits, (0, missing), 'constant')
    else:
        excess = len(rx_bits) - len(bit_stream)
        percent_excess = (excess / len(bit_stream)) * 100
        print(f"Excess {excess} bits ({percent_excess:.2f}% of original)")
        rx_bits = rx_bits[:len(bit_stream)]

    # Convert back to image
    rx_bytes = np.packbits(rx_bits)
    rx_img_array = rx_bytes.reshape(original_shape)
    rx_image = Image.fromarray(rx_img_array)
    rx_image.save("results/debug_ofdm_received.png")

    return berZ


def main():
    print("\n=== Running Improved ML Image Transmission System ===")

    print("\n=== Running Basic Validation Tests ===")
    simple_ber = test_simple_image_transmission()
    direct_ber, direct_quality = test_direct_image_transmission()
    ofdm_debug_ber = debug_ofdm_transmission()

    print("\n=== Basic Test Results ===")
    print(f"Simple test BER: {simple_ber:.6f}")
    print(f"Direct test BER: {direct_ber:.6f}, PSNR: {direct_quality['psnr']:.2f}")
    print(f"OFDM debug BER: {ofdm_debug_ber:.6f}")

    # Test basic OFDM
    ofdm_ber = test_ofdm(snr_db=25)

    # Test beamforming and channel
    bf_ber = test_beamforming_channel(snr_db=25)

    # Test adaptive modulation
    adaptive_results = test_adaptive_modulation()

    # Test different preprocessing and encoding strategies
    test_strategies = [
        {"name": "Basic BPSK", "modulation": "BPSK", "use_ldpc": False, "use_crc": False, "size": (128, 128)},
        {"name": "BPSK+CRC", "modulation": "BPSK", "use_ldpc": False, "use_crc": True, "size": (128, 128)},
        {"name": "BPSK+LDPC", "modulation": "BPSK", "use_ldpc": True, "use_crc": False, "size": (128, 128)},
        {"name": "Traditional Adaptive", "modulation": "traditional", "use_ldpc": False, "use_crc": True, "size": (64, 64)},
        {"name": "ML Adaptive", "modulation": "ml_adaptive", "use_ldpc": False, "use_crc": True, "size": (64, 64)},
        {"name": "ML Adaptive+LDPC", "modulation": "auto", "use_ldpc": True, "use_crc": True, "size": (64, 64)}
    ]

    results = []
    for strategy in test_strategies:
        print(f"\n--- Testing {strategy['name']} ---")
        ber, quality, _ = test_full_system(
            image_size=strategy["size"],
            modulation=strategy["modulation"],
            use_ldpc=strategy["use_ldpc"],
            use_crc=strategy["use_crc"],
            snr_db=25
        )
        results.append({
            'name': strategy['name'],
            'ber': ber,
            'psnr': quality['psnr'],
            'ssim': quality['ssim']
        })

    # Add the new test here, after processing all the test strategies
    # Compare ML and traditional adaptive modulation
    print("\n=== Running Adaptive Modulation Comparison ===")
    trad_results, ml_results = test_adaptive_comparison(snr_range=[15, 20, 25, 30])

    # Generate LDPC performance visualization
    try:
        ldpc_vis_file = visualize_ldpc_performance()
        print(f"LDPC performance visualization saved to: {ldpc_vis_file}")
    except Exception as e:
        print(f"Could not generate LDPC visualization: {str(e)}")

    # Print results comparison
    print("\n=== Image Transmission Strategy Comparison ===")
    print(f"{'Strategy':<25} {'BER':<10} {'PSNR (dB)':<15} {'SSIM':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<25} {r['ber']:<10.6f} {r['psnr']:<15.2f} {r['ssim']:<10.4f}")

    # Create a summary plot with all results
    plt.figure(figsize=(12, 10))

    # Plot BER comparison
    plt.subplot(2, 1, 1)
    names = [r['name'] for r in results]
    bers = [r['ber'] for r in results]
    psnrs = [r['psnr'] for r in results]

    x = np.arange(len(names))
    bar_width = 0.35

    plt.bar(x, bers, width=bar_width, label='BER (lower is better)')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Bit Error Rate')
    plt.title('Comparison of Transmission Strategies - BER')
    plt.grid(axis='y')

    # Plot PSNR comparison
    plt.subplot(2, 1, 2)
    plt.bar(x, psnrs, width=bar_width, color='green', label='PSNR (higher is better)')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('PSNR (dB)')
    plt.title('Comparison of Transmission Strategies - PSNR')
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig("results/strategy_comparison.png")
    plt.close()

    print("\nSummary plots saved to results/strategy_comparison.png")
    print("ML training visualization saved to results/ml_training_results.png")


if __name__ == "__main__":
    main()