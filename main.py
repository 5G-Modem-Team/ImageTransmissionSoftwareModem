#!/usr/bin/env python3
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import argparse

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

from transmitter import Transmitter
from receiver import Receiver
from channel import Channel
from ofdm import OFDMProcessor, OFDMBeamformer
from adaptive_mod import AdaptiveModulation
from utils import generate_test_image, calculate_psnr, plot_radiation_pattern, analyze_image_quality


class ImageTransmissionSystem:
    def __init__(self, num_tx_antennas=4, num_rx_antennas=4):
        self.num_tx_antennas = num_tx_antennas
        self.num_rx_antennas = num_rx_antennas
        self.figure_counter = 1

    def save_plot(self, prefix, output_dir='results'):
        """Save plot to file"""
        filename = os.path.join(output_dir, f"{prefix}_{self.figure_counter}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.figure_counter += 1
        return filename

    def image_to_bits(self, image):
        """Convert image to bit stream"""
        if isinstance(image, str):
            img = Image.open(image).convert('L')
        else:
            img = image.convert('L')
        # Resize to ensure dimensions are multiples of OFDM parameters
        new_size = (256, 256)  # Adjust size to work with OFDM parameters
        img = img.resize(new_size)
        img_array = np.array(img)
        bit_stream = np.unpackbits(img_array.astype(np.uint8))
        return bit_stream, img_array.shape, img_array

    def bits_to_image(self, bits, shape):
        """Convert bit stream back to image"""
        # Ensure bits length matches expected size
        expected_bits = shape[0] * shape[1] * 8
        if len(bits) > expected_bits:
            bits = bits[:expected_bits]
        elif len(bits) < expected_bits:
            bits = np.pad(bits, (0, expected_bits - len(bits)))

        bytes_data = np.packbits(bits)
        img_array = bytes_data.reshape(shape)
        return Image.fromarray(img_array), img_array

    def simulate_transmission(self, image, modulation="QPSK", snr_db=20,
                              beam_angle=30, use_ofdm=True, use_adaptive_mod=True,
                              output_dir='results'):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Convert image to bits
        input_bits, original_shape, original_array = self.image_to_bits(image)

        # Initialize components
        tx = Transmitter(input_bits, modulation=modulation, num_antennas=self.num_tx_antennas)
        channel = Channel(self.num_tx_antennas, self.num_rx_antennas)
        rx = Receiver(modulation=modulation, num_antennas=self.num_rx_antennas)

        # Initialize adaptive modulation if enabled
        if use_adaptive_mod:
            adaptive_mod = AdaptiveModulation()
            channel_state = {
                'H': channel.get_channel_matrix(),
                'noise_variance': 10 ** (-snr_db / 10)
            }
            adaptation_result = adaptive_mod.adapt_to_channel(
                channel_state['H'],
                channel_state['noise_variance']
            )
            print(f"Selected modulation: {adaptation_result['base_modulation'].name}")
            print(f"Channel capacity: {adaptation_result['channel_capacity']:.2f} bits/s/Hz")
            modulation = adaptation_result['base_modulation'].name

        # Transmit with beamforming
        print(f"Transmitting with beamforming at {beam_angle} degrees...")
        tx_symbols = tx.transmit(beam_angle=beam_angle)

        # Save transmit constellation
        plt.figure(figsize=(8, 8))
        tx.constellation(tx_symbols[:, 0] if len(tx_symbols.shape) > 1 else tx_symbols)
        self.save_plot('tx_constellation', output_dir)

        # Apply OFDM if enabled
        if use_ofdm:
            ofdm = OFDMProcessor()
            ofdm_beamformer = OFDMBeamformer(self.num_tx_antennas)
            print("Applying OFDM modulation...")

            # Ensure signal length is compatible with OFDM parameters
            num_symbols = len(tx_symbols)
            if len(tx_symbols.shape) > 1:
                tx_symbols = tx_symbols[:, 0]

            # Pad signal if necessary
            pad_length = (-len(tx_symbols)) % ofdm.nfft
            if pad_length:
                tx_symbols = np.pad(tx_symbols, (0, pad_length))

            ofdm_signal = ofdm.modulate(tx_symbols)
            beamformed_ofdm = ofdm_beamformer.apply_beamforming(ofdm_signal, beam_angle)
            tx_symbols = beamformed_ofdm

        # Plot and save radiation pattern
        if beam_angle is not None:
            plt.figure(figsize=(10, 6))
            tx.plot_radiation_pattern(tx_symbols)
            self.save_plot('radiation_pattern', output_dir)

        # Transmit through channel
        rx_symbols = channel.apply_channel(tx_symbols, snr_db)

        # Receive and demodulate
        rx_bits, eq_symbols = rx.receive(rx_symbols, channel.get_channel_matrix())

        # Save receive constellation
        plt.figure(figsize=(8, 8))
        rx.constellation_plot(eq_symbols, f"Received {modulation} Constellation")
        self.save_plot('rx_constellation', output_dir)

        # Calculate BER
        ber = rx.calculate_ber(input_bits[:len(rx_bits)], rx_bits[:len(input_bits)])
        print(f"Bit Error Rate: {ber:.6f}")

        # Convert back to image
        received_image, received_array = self.bits_to_image(rx_bits[:len(input_bits)], original_shape)

        # Calculate image quality metrics
        quality_metrics = analyze_image_quality(original_array, received_array)
        print("\nImage Quality Metrics:")
        for metric, value in quality_metrics.items():
            print(f"{metric}: {value:.2f}")

        return received_image, ber, quality_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Image Transmission Software Modem')
    parser.add_argument('--image', type=str, default=None,
                        help='Input image path (generates test image if not provided)')
    parser.add_argument('--modulation', type=str, default='QPSK',
                        choices=['BPSK', 'QPSK', '16QAM', '64QAM'],
                        help='Modulation scheme')
    parser.add_argument('--snr', type=float, default=20,
                        help='Signal-to-Noise Ratio in dB')
    parser.add_argument('--beam-angle', type=float, default=30,
                        help='Beamforming angle in degrees')
    parser.add_argument('--use-ofdm', action='store_true',
                        help='Enable OFDM')
    parser.add_argument('--use-adaptive', action='store_true',
                        help='Enable adaptive modulation')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create system instance
    system = ImageTransmissionSystem(num_tx_antennas=4, num_rx_antennas=4)

    # Load or generate test image
    if args.image:
        input_image = Image.open(args.image).convert('L')
        print(f"Loaded image from {args.image}")
    else:
        print("Generating test image...")
        input_image = generate_test_image()

    # Save original image
    input_image.save(os.path.join(args.output_dir, 'original.png'))
    print(f"Original image saved as '{os.path.join(args.output_dir, 'original.png')}'")

    # Test cases
    test_cases = [
        {
            'name': 'Basic Transmission',
            'params': {
                'modulation': args.modulation,
                'snr_db': args.snr,
                'beam_angle': args.beam_angle,
                'use_ofdm': False,
                'use_adaptive_mod': False,
                'output_dir': args.output_dir
            }
        },
        {
            'name': 'Beamforming + OFDM',
            'params': {
                'modulation': args.modulation,
                'snr_db': args.snr,
                'beam_angle': args.beam_angle,
                'use_ofdm': True,
                'use_adaptive_mod': False,
                'output_dir': args.output_dir
            }
        },
        {
            'name': 'Adaptive Modulation',
            'params': {
                'modulation': args.modulation,
                'snr_db': args.snr,
                'beam_angle': args.beam_angle,
                'use_ofdm': True,
                'use_adaptive_mod': True,
                'output_dir': args.output_dir
            }
        }
    ]

    # Create figure for final results
    plt.figure(figsize=(15, 10))

    # Show original image
    plt.subplot(241)
    plt.imshow(input_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        received_image, ber, quality_metrics = system.simulate_transmission(input_image, **test_case['params'])
        results.append((received_image, ber, quality_metrics))

        # Add to results plot
        plt.subplot(2, 4, i + 1)
        plt.imshow(received_image, cmap='gray')
        plt.title(f"{test_case['name']}\nBER: {ber:.6f}\nPSNR: {quality_metrics['psnr']:.1f} dB")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'transmission_results.png'), dpi=300)
    plt.close()

    print("\nAll results have been saved in the output directory:")
    print(f"- {args.output_dir}/transmission_results.png : Overall comparison")
    print(f"- {args.output_dir}/tx_constellation_*.png : Transmit constellations")
    print(f"- {args.output_dir}/rx_constellation_*.png : Receive constellations")
    print(f"- {args.output_dir}/radiation_pattern_*.png : Beamforming radiation patterns")


if __name__ == "__main__":
    main()