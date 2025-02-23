import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv


class Receiver:
    def __init__(self, modulation="QPSK", num_antennas=4):
        self.modulation = modulation
        self.num_antennas = num_antennas

    def zf_equalizer(self, received_signal, channel_matrix):
        """Zero-Forcing equalizer for MIMO systems"""
        # Ensure received_signal is 2D
        if len(received_signal.shape) == 3:
            # For OFDM case, process each time sample
            equalized = np.zeros_like(received_signal)
            for i in range(received_signal.shape[0]):
                for j in range(received_signal.shape[1]):
                    H_pinv = pinv(channel_matrix)
                    equalized[i, j, :] = np.dot(received_signal[i, j, :], H_pinv.T)
            return equalized
        else:
            # Calculate pseudo-inverse of channel matrix
            H_pinv = pinv(channel_matrix)
            # Apply equalizer
            equalized_signal = np.dot(received_signal, H_pinv.T)
            return equalized_signal

    def demodulate_BPSK(self, symbols):
        """Demodulate BPSK symbols"""
        return (np.real(symbols) > 0).astype(int)

    def demodulate_QPSK(self, symbols):
        """Demodulate QPSK symbols"""
        # Flatten array if it's multi-dimensional
        symbols = symbols.flatten()
        # Get real and imaginary components
        real_bits = (np.real(symbols) > 0).astype(int)
        imag_bits = (np.imag(symbols) > 0).astype(int)
        # Interleave real and imaginary bits
        bits = np.zeros(2 * len(symbols), dtype=int)
        bits[0::2] = real_bits
        bits[1::2] = imag_bits
        return bits

    def demodulate_QAM16(self, symbols):
        """Demodulate 16-QAM symbols"""
        # Flatten array if it's multi-dimensional
        symbols = symbols.flatten()

        # Scale symbol back
        symbols = symbols * np.sqrt(10)

        # Get real and imaginary parts
        real_parts = np.real(symbols)
        imag_parts = np.imag(symbols)

        # Decode real parts (2 bits per real part)
        real_bits1 = (real_parts > 0).astype(int)
        real_bits2 = (np.abs(real_parts) > 2).astype(int)

        # Decode imaginary parts (2 bits per imaginary part)
        imag_bits1 = (imag_parts > 0).astype(int)
        imag_bits2 = (np.abs(imag_parts) > 2).astype(int)

        # Combine bits in correct order
        bits = np.zeros(4 * len(symbols), dtype=int)
        bits[0::4] = real_bits1
        bits[1::4] = real_bits2
        bits[2::4] = imag_bits1
        bits[3::4] = imag_bits2

        return bits

    def demodulate_QAM64(self, symbols):
        """Demodulate 64-QAM symbols"""
        # Flatten array if it's multi-dimensional
        symbols = symbols.flatten()

        # Scale symbol back
        symbols = symbols * np.sqrt(42)

        # Get real and imaginary parts
        real_parts = np.real(symbols)
        imag_parts = np.imag(symbols)

        # Decode real parts (3 bits per real part)
        real_bits1 = (real_parts > 0).astype(int)
        abs_real = np.abs(real_parts)
        real_bits2 = (abs_real > 2).astype(int)
        real_bits3 = (abs_real % 2 > 1).astype(int)

        # Decode imaginary parts (3 bits per imaginary part)
        imag_bits1 = (imag_parts > 0).astype(int)
        abs_imag = np.abs(imag_parts)
        imag_bits2 = (abs_imag > 2).astype(int)
        imag_bits3 = (abs_imag % 2 > 1).astype(int)

        # Combine bits in correct order
        bits = np.zeros(6 * len(symbols), dtype=int)
        bits[0::6] = real_bits1
        bits[1::6] = real_bits2
        bits[2::6] = real_bits3
        bits[3::6] = imag_bits1
        bits[4::6] = imag_bits2
        bits[5::6] = imag_bits3

        return bits

    def calculate_ber(self, original_bits, received_bits):
        """Calculate Bit Error Rate"""
        min_len = min(len(original_bits), len(received_bits))
        return np.mean(original_bits[:min_len] != received_bits[:min_len])

    def constellation_plot(self, symbols, title="Received Constellation"):
        """Plot constellation diagram of received symbols"""
        # If symbols is 3D array (OFDM case), take first symbol
        if len(symbols.shape) > 2:
            symbols = symbols[0, :, 0]
        elif len(symbols.shape) > 1:
            symbols = symbols[:, 0]

        plt.figure(figsize=(8, 8))
        plt.plot(np.real(symbols), np.imag(symbols), '.')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.title(title)
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.grid(True)

    def receive(self, received_signal, channel_matrix=None):
        """Process received signal and demodulate"""
        # Apply equalizer if channel matrix is provided
        if channel_matrix is not None:
            equalized_signal = self.zf_equalizer(received_signal, channel_matrix)
        else:
            equalized_signal = received_signal

        # Demodulate based on modulation scheme
        if self.modulation == "BPSK":
            bits = self.demodulate_BPSK(equalized_signal)
        elif self.modulation == "QPSK":
            bits = self.demodulate_QPSK(equalized_signal)
        elif self.modulation == "16QAM":
            bits = self.demodulate_QAM16(equalized_signal)
        elif self.modulation == "64QAM":
            bits = self.demodulate_QAM64(equalized_signal)
        else:
            raise ValueError(f"Unsupported modulation scheme: {self.modulation}")

        return bits, equalized_signal


if __name__ == "__main__":
    # Test the receiver
    # Generate test data
    num_symbols = 1000
    original_bits = np.random.randint(0, 2, num_symbols * 2)  # For QPSK

    # Create test signal (QPSK)
    test_symbols = np.array([(original_bits[i] * 2 - 1) + 1j * (original_bits[i + 1] * 2 - 1)
                             for i in range(0, len(original_bits), 2)]) / np.sqrt(2)

    # Add some noise
    noisy_symbols = test_symbols + 0.1 * (np.random.randn(len(test_symbols)) +
                                          1j * np.random.randn(len(test_symbols)))

    # Create receiver
    rx = Receiver(modulation="QPSK")

    # Receive and demodulate
    received_bits, equalized_symbols = rx.receive(noisy_symbols)

    # Calculate BER
    ber = rx.calculate_ber(original_bits, received_bits)
    print(f"Bit Error Rate: {ber:.6f}")

    # Plot constellation
    rx.constellation_plot(equalized_symbols, "Received QPSK Constellation")