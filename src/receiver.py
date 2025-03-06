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
            equalized = np.zeros((received_signal.shape[0], received_signal.shape[1]), dtype=complex)
            for i in range(received_signal.shape[0]):
                for j in range(received_signal.shape[1]):
                    H_pinv = pinv(channel_matrix)
                    equalized[i, j] = np.dot(H_pinv[0, :], received_signal[i, j, :])
            return equalized
        else:
            # Check if we're dealing with a multi-antenna signal (MIMO case)
            if len(received_signal.shape) > 1 and received_signal.shape[1] > 1:
                # Calculate pseudo-inverse of channel matrix
                H_pinv = pinv(channel_matrix)
                # Apply equalizer (transpose properly for matrix multiplication)
                equalized_signal = np.zeros(received_signal.shape[0], dtype=complex)
                for i in range(received_signal.shape[0]):
                    # Use the first row of H_pinv for simplicity
                    equalized_signal[i] = np.dot(H_pinv[0, :], received_signal[i, :])
                return equalized_signal.reshape(-1, 1)
            else:
                # For SISO case
                return received_signal

    def demodulate_BPSK(self, symbols):
        """Demodulate BPSK symbols"""
        # Ensure symbols is flattened for consistent processing
        symbols_flat = symbols.flatten()
        return (np.real(symbols_flat) > 0).astype(int)

    def demodulate_QPSK(self, symbols):
        """Demodulate QPSK symbols"""
        # Ensure symbols is flattened for consistent processing
        symbols_flat = symbols.flatten()

        # Get real and imaginary components
        real_bits = (np.real(symbols_flat) > 0).astype(int)
        imag_bits = (np.imag(symbols_flat) > 0).astype(int)

        # Interleave real and imaginary bits
        bits = np.zeros(2 * len(symbols_flat), dtype=int)
        bits[0::2] = real_bits
        bits[1::2] = imag_bits
        return bits

    def demodulate_QAM64(self, symbols):
        """Demodulate 64-QAM symbols"""
        # Ensure symbols is flattened
        symbols_flat = symbols.flatten()

        # Scale symbol back
        symbols_flat = symbols_flat * np.sqrt(42)

        # Get real and imaginary parts
        real_parts = np.real(symbols_flat)
        imag_parts = np.imag(symbols_flat)

        # Map to bit values (inverse of transmitter mapping)
        # Quantize to nearest valid level
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        real_indices = np.argmin(np.abs(real_parts.reshape(-1, 1) - levels), axis=1)
        imag_indices = np.argmin(np.abs(imag_parts.reshape(-1, 1) - levels), axis=1)

        # Convert indices to 3-bit binary representation
        real_bits = ((real_indices.reshape(-1, 1) & (1 << np.arange(3))) > 0).astype(int)[:, ::-1]
        imag_bits = ((imag_indices.reshape(-1, 1) & (1 << np.arange(3))) > 0).astype(int)[:, ::-1]

        # Combine all bits
        all_bits = np.column_stack((real_bits, imag_bits)).flatten()

        return all_bits

    def demodulate_QAM16(self, symbols):
        """Demodulate 16-QAM symbols"""
        # Ensure symbols is flattened
        symbols_flat = symbols.flatten()

        # Scale symbol back
        symbols_flat = symbols_flat * np.sqrt(10)

        # Get real and imaginary parts
        real_parts = np.real(symbols_flat)
        imag_parts = np.imag(symbols_flat)

        # Map to bit values (inverse of transmitter mapping)
        # For real part: first bit is sign, second bit is magnitude
        real_bits1 = (real_parts > 0).astype(int)
        real_bits2 = (np.abs(real_parts) > 2).astype(int)

        # For imaginary part: first bit is sign, second bit is magnitude
        imag_bits1 = (imag_parts > 0).astype(int)
        imag_bits2 = (np.abs(imag_parts) > 2).astype(int)

        # Combine bits in correct order
        bits = np.zeros(4 * len(symbols_flat), dtype=int)
        bits[0::4] = real_bits1
        bits[1::4] = real_bits2
        bits[2::4] = imag_bits1
        bits[3::4] = imag_bits2

        return bits

    def calculate_ber(self, original_bits, received_bits):
        """Calculate Bit Error Rate"""
        min_len = min(len(original_bits), len(received_bits))
        return np.mean(original_bits[:min_len] != received_bits[:min_len])

    def constellation_plot(self, symbols, title="Received Constellation"):
        """Plot constellation diagram of received symbols"""
        # Flatten or extract symbols for plotting
        if len(symbols.shape) > 2:
            # OFDM case
            symbols_plot = symbols[0, :, 0]
        elif len(symbols.shape) > 1 and symbols.shape[1] > 1:
            # MIMO case, use first antenna
            symbols_plot = symbols[:, 0]
        else:
            # Already flattened or single antenna
            symbols_plot = symbols.flatten()

        plt.figure(figsize=(8, 8))
        plt.plot(np.real(symbols_plot), np.imag(symbols_plot), '.')
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
        mod_upper = self.modulation.upper()

        if mod_upper == "BPSK":
            bits = self.demodulate_BPSK(equalized_signal)
        elif mod_upper == "QPSK":
            bits = self.demodulate_QPSK(equalized_signal)
        elif mod_upper in ["16QAM", "QAM16"]:
            bits = self.demodulate_QAM16(equalized_signal)
        elif mod_upper in ["64QAM", "QAM64"]:
            bits = self.demodulate_QAM64(equalized_signal)
        else:
            raise ValueError(f"Unsupported modulation scheme: {self.modulation}")

        return bits, equalized_signal