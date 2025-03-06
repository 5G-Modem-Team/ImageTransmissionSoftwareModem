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
        # Ensure even number of bits
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)

        # Create QPSK symbols
        symbols = []
        for i in range(0, len(bits), 2):
            real = 2 * bits[i] - 1
            imag = 2 * bits[i + 1] - 1
            symbols.append(complex(real, imag))

        # Normalize to unit power
        symbols = np.array(symbols) / np.sqrt(2)
        self.symbols = symbols
        return self.symbols

    def QAM16(self, bits):
        self.modulation = "16QAM"
        # Ensure bits length is multiple of 4
        if len(bits) % 4 != 0:
            bits = np.pad(bits, (0, 4 - (len(bits) % 4)), 'constant')

        symbols = []
        for i in range(0, len(bits), 4):
            # Handle case where we might be at the end of the array
            if i + 4 > len(bits):
                bit_group = np.pad(bits[i:], (0, 4 - (len(bits) - i)), 'constant')
            else:
                bit_group = bits[i:i + 4]

            real_part = (2 * bit_group[0] - 1) * (2 + (2 * bit_group[1] - 1))
            imag_part = (2 * bit_group[2] - 1) * (2 + (2 * bit_group[3] - 1))
            symbols.append(complex(real_part, imag_part))

        symbols = np.array(symbols)
        self.symbols = symbols / np.sqrt(10)
        return self.symbols

    def QAM64(self, bits):
        self.modulation = "64QAM"
        # Ensure number of bits is multiple of 6
        if len(bits) % 6 != 0:
            bits = np.pad(bits, (0, 6 - (len(bits) % 6)), 'constant')

        symbols = []
        for i in range(0, len(bits), 6):
            # Handle case where we might be at the end of the array
            if i + 6 > len(bits):
                bit_group = np.pad(bits[i:], (0, 6 - (len(bits) - i)), 'constant')
            else:
                bit_group = bits[i:i + 6]

            # Extract 3 bits for real and 3 bits for imaginary
            real_bits = bit_group[0:3]
            imag_bits = bit_group[3:6]

            # Map 3 bits to 8 possible levels (-7, -5, -3, -1, 1, 3, 5, 7)
            real_val = (4 * real_bits[0] + 2 * real_bits[1] + real_bits[2]) * 2 - 7
            imag_val = (4 * imag_bits[0] + 2 * imag_bits[1] + imag_bits[2]) * 2 - 7

            symbols.append(complex(real_val, imag_val))

        # Normalize to unit average power
        symbols = np.array(symbols) / np.sqrt(42)  # 42 is average power of 64QAM
        self.symbols = symbols
        return self.symbols

    def apply_beamforming(self, symbols, angle_degrees):
        """Apply beamforming weights for desired transmission angle"""
        angle_rad = np.deg2rad(angle_degrees)
        d = 0.5  # Half wavelength antenna spacing
        k = 2 * np.pi
        n = np.arange(self.num_antennas)

        # Calculate steering vector
        steering_vector = np.exp(1j * k * d * n * np.sin(angle_rad))

        # Normalize steering vector
        steering_vector = steering_vector / np.sqrt(self.num_antennas)

        # Handle different input dimensions
        if len(symbols.shape) == 1:
            # For 1D symbols, apply beamforming
            beamformed_signals = np.zeros((len(symbols), self.num_antennas), dtype=complex)
            for i in range(len(symbols)):
                beamformed_signals[i, :] = symbols[i] * steering_vector
        else:
            # For 2D case (e.g. OFDM)
            beamformed_signals = np.zeros((symbols.shape[0], symbols.shape[1], self.num_antennas), dtype=complex)
            for i in range(symbols.shape[0]):
                for j in range(symbols.shape[1]):
                    beamformed_signals[i, j, :] = symbols[i, j] * steering_vector

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
        mod_upper = self.modulation.upper()

        if mod_upper == "BPSK":
            symbols = self.BPSK(self.bits)
        elif mod_upper == "QPSK":
            symbols = self.QPSK(self.bits)
        elif mod_upper in ["16QAM", "QAM16"]:
            symbols = self.QAM16(self.bits)
        elif mod_upper in ["64QAM", "QAM64"]:
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