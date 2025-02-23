import numpy as np
import matplotlib.pyplot as plt


class Transmitter:
    def __init__(self, bits, modulation="QPSK", num_antennas=4):
        self.bits = bits
        self.modulation = modulation
        self.symbols = None
        self.num_antennas = num_antennas

    def BPSK(self, bits):
        self.modulation = "BPSK"
        self.symbols = bits * 2 - 1
        return self.symbols

    def QPSK(self, bits):
        self.modulation = "QPSK"
        symbols = np.array([(bits[i] * 2 - 1) + 1j * (bits[i + 1] * 2 - 1)
                            for i in range(0, len(bits), 2)])
        magnitude = abs(1 + 1j)
        symbols_normalized = symbols / magnitude
        self.symbols = symbols_normalized
        return self.symbols

    def QAM16(self, bits):
        self.modulation = "16QAM"
        symbols = []
        for i in range(0, len(bits), 4):
            bit_group = bits[i:i + 4]
            real_part = (2 * bit_group[0] - 1) * (2 + (2 * bit_group[1] - 1))
            imag_part = (2 * bit_group[2] - 1) * (2 + (2 * bit_group[3] - 1))
            symbols.append(complex(real_part, imag_part))

        symbols = np.array(symbols)
        self.symbols = symbols / np.sqrt(10)
        return self.symbols

    def QAM64(self, bits):
        self.modulation = "64QAM"
        symbols = []
        for i in range(0, len(bits), 6):
            bit_group = bits[i:i + 6]
            real_bits = bit_group[0:3]
            imag_bits = bit_group[3:6]

            real_part = (4 * real_bits[0] - 2) + (2 * real_bits[1] - 1) + (real_bits[2] - 0.5)
            imag_part = (4 * imag_bits[0] - 2) + (2 * imag_bits[1] - 1) + (imag_bits[2] - 0.5)

            symbols.append(complex(real_part, imag_part))

        symbols = np.array(symbols)
        self.symbols = symbols / np.sqrt(42)
        return self.symbols

    def apply_beamforming(self, symbols, angle_degrees):
        """Apply beamforming weights for desired transmission angle"""
        angle_rad = np.deg2rad(angle_degrees)
        d = 0.5
        k = 2 * np.pi
        n = np.arange(self.num_antennas)
        steering_vector = np.exp(1j * k * d * n * np.sin(angle_rad))
        steering_vector = steering_vector / np.sqrt(self.num_antennas)
        beamformed_signals = np.outer(symbols, steering_vector)
        return beamformed_signals

    def plot_radiation_pattern(self, weights):
        """Plot radiation pattern for given beamforming weights"""
        # If weights is 3D array (OFDM case), take first symbol's first carrier
        if len(weights.shape) > 2:
            weights = weights[0, 0, :]
        elif len(weights.shape) > 1:
            weights = weights[0, :]

        angles = np.linspace(-90, 90, 360)
        pattern = []

        for angle in angles:
            angle_rad = np.deg2rad(angle)
            d = 0.5
            k = 2 * np.pi
            n = np.arange(self.num_antennas)
            steering_vector = np.exp(1j * k * d * n * np.sin(angle_rad))
            # Add small constant to avoid log10(0)
            response = np.abs(np.dot(weights, steering_vector)) ** 2 + 1e-10
            pattern.append(response)

        pattern = np.array(pattern)

        plt.figure(figsize=(10, 6))
        plt.plot(angles, 10 * np.log10(pattern))
        plt.grid(True)
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Relative Power (dB)')
        plt.title('Radiation Pattern')
        plt.ylim(-40, 10)  # Set reasonable y-axis limits for dB scale

    def constellation(self, symbols):
        # If symbols is 3D array (OFDM case), take first symbol
        if len(symbols.shape) > 2:
            symbols = symbols[0, :, 0]
        elif len(symbols.shape) > 1:
            symbols = symbols[:, 0]

        plt.figure(figsize=(8, 8))
        plt.plot(np.real(symbols), np.imag(symbols), '.')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.title(f"{self.modulation} Constellation Diagram")
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.grid(True)

    def transmit(self, beam_angle=None):
        # Modulate the bits
        if self.modulation == "BPSK":
            symbols = self.BPSK(self.bits)
        elif self.modulation == "QPSK":
            symbols = self.QPSK(self.bits)
        elif self.modulation == "16QAM":
            symbols = self.QAM16(self.bits)
        elif self.modulation == "64QAM":
            symbols = self.QAM64(self.bits)
        else:
            raise Exception(f"Modulation: {self.modulation} not supported")

        # Apply beamforming if angle is specified
        if beam_angle is not None:
            beamformed_signals = self.apply_beamforming(symbols, beam_angle)
            return beamformed_signals

        return symbols


if __name__ == "__main__":
    # Generate random bits
    num_bits = 1000
    bits = np.random.randint(0, 2, num_bits)

    # Test different modulations
    modulations = ["BPSK", "QPSK", "16QAM", "64QAM"]

    for mod in modulations:
        tx = Transmitter(bits=bits, modulation=mod)
        symbols = tx.transmit()
        tx.constellation(symbols)

    # Test beamforming
    tx = Transmitter(bits=bits, modulation="QPSK", num_antennas=4)
    beam_angle = 30  # degrees
    beamformed_signals = tx.transmit(beam_angle=beam_angle)

    # Plot radiation pattern
    weights = tx.apply_beamforming(np.array([1]), beam_angle)
    tx.plot_radiation_pattern(weights[0])