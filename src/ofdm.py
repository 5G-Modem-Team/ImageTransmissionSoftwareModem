import numpy as np


class OFDMProcessor:
    def __init__(self, nfft=64, ncp=16, ncarriers=52):
        self.nfft = nfft  # FFT size
        self.ncp = ncp  # Cyclic prefix length
        self.ncarriers = ncarriers  # Number of data carriers
        self.nsym = self.nfft + self.ncp  # Total symbol length

        # Generate carrier indices
        self.carrier_indices = np.concatenate([
            np.arange(-ncarriers // 2, 0),
            np.arange(1, ncarriers // 2 + 1)
        ]) + nfft // 2

    def modulate(self, symbols):
        """
        OFDM modulation
        Input: symbols to be transmitted on subcarriers
        Output: OFDM modulated time domain signal
        """
        # Calculate number of OFDM symbols needed
        n_symbols = (len(symbols) + self.ncarriers - 1) // self.ncarriers

        # Pad the input symbols if necessary
        padded_length = n_symbols * self.ncarriers
        if len(symbols) < padded_length:
            symbols = np.pad(symbols, (0, padded_length - len(symbols)))

        # Reshape input for OFDM symbols
        symbols = symbols.reshape(n_symbols, -1)

        # Prepare OFDM symbols in frequency domain
        ofdm_symbols = np.zeros((n_symbols, self.nfft), dtype=complex)
        ofdm_symbols[:, self.carrier_indices] = symbols[:, :len(self.carrier_indices)]

        # IFFT to convert to time domain
        time_signal = np.fft.ifft(ofdm_symbols, axis=1) * np.sqrt(self.nfft)

        # Add cyclic prefix
        cp = time_signal[:, -self.ncp:]
        ofdm_signal = np.hstack([cp, time_signal])

        return ofdm_signal.flatten()

    def demodulate(self, received_signal):
        """
        OFDM demodulation
        Input: received time domain signal
        Output: demodulated symbols from subcarriers
        """
        # Calculate number of complete OFDM symbols
        n_symbols = len(received_signal) // self.nsym

        # Truncate signal to complete symbols
        received_signal = received_signal[:n_symbols * self.nsym]

        # Reshape into OFDM symbols
        received_signal = received_signal.reshape(n_symbols, self.nsym)

        # Remove cyclic prefix
        received_signal = received_signal[:, self.ncp:]

        # FFT to convert back to frequency domain
        freq_signal = np.fft.fft(received_signal, axis=1) / np.sqrt(self.nfft)

        # Extract data from subcarriers
        received_symbols = freq_signal[:, self.carrier_indices]

        return received_symbols.flatten()


class OFDMBeamformer:
    def __init__(self, n_antennas, nfft=64):
        self.n_antennas = n_antennas
        self.nfft = nfft

    def calculate_steering_vector(self, angle_degrees, freq_idx):
        """Calculate steering vector for specific frequency and angle"""
        angle_rad = np.deg2rad(angle_degrees)
        freq_norm = (freq_idx - self.nfft // 2) / self.nfft
        n = np.arange(self.n_antennas)
        phase_shifts = 2 * np.pi * freq_norm * n * np.sin(angle_rad)
        return np.exp(1j * phase_shifts)

    def apply_beamforming(self, ofdm_signal, angle_degrees):
        """Apply beamforming weights to OFDM symbols"""
        # Calculate number of OFDM symbols
        nsym = self.nfft + 16  # OFDM symbol length (FFT size + CP)
        n_symbols = len(ofdm_signal) // nsym

        # Reshape signal into OFDM symbols
        symbols = ofdm_signal[:n_symbols * nsym].reshape(n_symbols, nsym)

        # Initialize output array for all antennas
        beamformed = np.zeros((n_symbols, nsym, self.n_antennas), dtype=complex)

        # Apply beamforming for each time sample
        for i in range(n_symbols):
            steering_vector = self.calculate_steering_vector(angle_degrees, i % self.nfft)
            beamformed[i, :, :] = np.outer(symbols[i, :], steering_vector)

        return beamformed