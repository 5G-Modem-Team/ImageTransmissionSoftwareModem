import numpy as np
import matplotlib.pyplot as plt


class Channel:
    def __init__(self, num_tx_antennas=4, num_rx_antennas=4,
                 channel_type='rayleigh', seed=None):
        """
        Initialize a wireless channel model

        Args:
            num_tx_antennas: Number of transmit antennas
            num_rx_antennas: Number of receive antennas
            channel_type: Type of channel ('rayleigh', 'rician', 'awgn')
            seed: Random seed for reproducibility
        """
        self.num_tx_antennas = num_tx_antennas
        self.num_rx_antennas = num_rx_antennas
        self.channel_type = channel_type
        self.H = None  # Channel matrix

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        self.generate_channel_matrix()

    def generate_channel_matrix(self):
        """Generate MIMO channel matrix based on selected channel type"""
        if self.channel_type == 'rayleigh':
            # Rayleigh fading channel - no line of sight component
            self.H = (np.random.randn(self.num_rx_antennas, self.num_tx_antennas) +
                      1j * np.random.randn(self.num_rx_antennas, self.num_tx_antennas)) / np.sqrt(2)
        elif self.channel_type == 'rician':
            # Rician fading channel - with line of sight component
            k_factor = 3  # Rician K-factor (ratio of LOS to scattered power)

            # Line of sight component
            los_component = np.ones((self.num_rx_antennas, self.num_tx_antennas))

            # Scattered component (Rayleigh)
            scattered = (np.random.randn(self.num_rx_antennas, self.num_tx_antennas) +
                         1j * np.random.randn(self.num_rx_antennas, self.num_tx_antennas)) / np.sqrt(2)

            # Combine LOS and scattered components
            self.H = np.sqrt(k_factor / (k_factor + 1)) * los_component + \
                     np.sqrt(1 / (k_factor + 1)) * scattered
        else:  # AWGN channel
            # Identity channel matrix (no fading)
            self.H = np.eye(self.num_rx_antennas, self.num_tx_antennas)

    def add_awgn(self, signal, snr_db, plot=False):
        """
        Add Additive White Gaussian Noise to the signal with reduced noise power

        Args:
            signal: Input signal
            snr_db: Signal-to-Noise Ratio in dB

        Returns:
            Noisy signal
        """
        # Calculate signal power accurately
        signal_power = np.mean(np.abs(signal) ** 2)

        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate complex noise
        if isinstance(signal, np.ndarray):
            noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) +
                                                1j * np.random.randn(*signal.shape))
        else:
            # Handle scalar case
            noise = np.sqrt(noise_power / 2) * (np.random.randn() + 1j * np.random.randn())

        if plot:
            fig, axs = plt.subplots(2)
            fig.suptitle(f"AWGN Noise at {snr_db}dB SNR")
            axs[0].plot(np.real(noise))
            axs[1].plot(np.imag(noise))
            axs[0].set_ylabel("Real Magnitude")
            axs[1].set_ylabel("Imaginary Magnitude")
            axs[1].set_xlabel("Bit Index")
            fig.savefig(f"results/awgn_noise_{snr_db}dB.png")
            plt.close(fig)

        return signal + noise

    def apply_channel(self, signal, snr_db):
        """
        Apply channel effects including fading and noise

        Args:
            signal: Input signal
            snr_db: Signal-to-Noise Ratio in dB

        Returns:
            Signal after channel effects
        """
        # Handle different input shapes
        if len(signal.shape) == 3:
            # OFDM with beamforming case: [time_samples, ofdm_symbol, antennas]
            output_signal = np.zeros((signal.shape[0], signal.shape[1], self.num_rx_antennas), dtype=complex)

            for t in range(signal.shape[0]):
                for s in range(signal.shape[1]):
                    # Apply channel matrix to each OFDM symbol
                    output_signal[t, s, :] = np.dot(self.H, signal[t, s, :])

            # Add noise
            output_signal = self.add_awgn(output_signal, snr_db)
            return output_signal

        elif len(signal.shape) == 2 and signal.shape[1] == self.num_tx_antennas:
            # Direct beamforming case: [time_samples, antennas]
            output_signal = np.zeros((signal.shape[0], self.num_rx_antennas), dtype=complex)

            for t in range(signal.shape[0]):
                # Apply channel matrix to each time sample
                output_signal[t, :] = np.dot(self.H, signal[t, :])

            # Add noise
            output_signal = self.add_awgn(output_signal, snr_db)
            return output_signal

        else:
            # Single antenna or other case
            # Convert to simple SISO if needed
            if self.num_tx_antennas == 1 and self.num_rx_antennas == 1:
                # Simple SISO channel
                channel_gain = np.abs(self.H[0, 0])
                output_signal = signal * channel_gain
            else:
                # Treat as first antenna only for simplicity
                output_signal = signal * self.H[0, 0]

            # Add noise
            output_signal = self.add_awgn(output_signal, snr_db)
            return output_signal

    def get_channel_matrix(self):
        """Return the current channel matrix"""
        return self.H
    

if __name__ == "__main__":
    # Example usage of the Channel class
    num_bits = 1000
    bits = np.random.randint(0, 2, num_bits)
    ch = Channel(num_tx_antennas=1,num_rx_antennas=1,channel_type='awgn',seed=42)
    ch.add_awgn(signal = bits, snr_db = 10, plot=True)