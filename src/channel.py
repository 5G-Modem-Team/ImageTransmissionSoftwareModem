import numpy as np


class Channel:
    def __init__(self, num_tx_antennas=4, num_rx_antennas=4):
        self.num_tx_antennas = num_tx_antennas
        self.num_rx_antennas = num_rx_antennas
        self.H = None  # Channel matrix
        self.generate_channel_matrix()

    def generate_channel_matrix(self):
        """Generate MIMO channel matrix with Rayleigh fading"""
        # Complex Gaussian random variables for Rayleigh fading
        self.H = (np.random.randn(self.num_rx_antennas, self.num_tx_antennas) +
                  1j * np.random.randn(self.num_rx_antennas, self.num_tx_antennas)) / np.sqrt(2)

    def add_awgn(self, signal, snr_db):
        """Add Additive White Gaussian Noise to the signal"""
        # Calculate signal power
        signal_power = np.mean(np.abs(signal) ** 2)

        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate complex noise
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) +
                                            1j * np.random.randn(*signal.shape))

        return signal + noise

    def apply_channel(self, signal, snr_db):
        """Apply channel effects including fading and noise"""
        # For MIMO transmission, signal should be of shape (num_symbols, num_tx_antennas)
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)

        # Apply channel matrix
        received_signal = np.dot(signal, self.H.T)

        # Add noise
        received_signal = self.add_awgn(received_signal, snr_db)

        return received_signal

    def get_channel_matrix(self):
        """Return the current channel matrix"""
        return self.H


if __name__ == "__main__":
    # Test the channel model
    num_symbols = 1000
    num_tx_antennas = 4
    num_rx_antennas = 4

    # Generate test signal
    test_signal = np.random.randn(num_symbols, num_tx_antennas) + 1j * np.random.randn(num_symbols, num_tx_antennas)

    # Create channel
    channel = Channel(num_tx_antennas, num_rx_antennas)

    # Apply channel effects
    snr_db = 20
    received_signal = channel.apply_channel(test_signal, snr_db)

    print(f"Original signal shape: {test_signal.shape}")
    print(f"Received signal shape: {received_signal.shape}")
    print(f"Channel matrix shape: {channel.get_channel_matrix().shape}")