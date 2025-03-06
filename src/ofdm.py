import numpy as np


class OFDMProcessor:
    def __init__(self, nfft=64, ncp=16, ncarriers=52):
        self.nfft = nfft  # FFT size
        self.ncp = ncp  # Cyclic prefix length
        self.ncarriers = ncarriers  # Number of data carriers
        self.nsym = self.nfft + self.ncp  # Total symbol length

        # Generate carrier indices (DC null and guard bands)
        self.carrier_indices = np.concatenate([
            np.arange(-ncarriers // 2, 0),
            np.arange(1, ncarriers // 2 + 1)
        ]) % nfft  # Use modulo to avoid negative indices

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
            symbols = np.pad(symbols, (0, padded_length - len(symbols)), 'constant')

        # Reshape input for OFDM symbols
        symbols = symbols.reshape(n_symbols, self.ncarriers)

        # Prepare OFDM symbols in frequency domain
        ofdm_symbols = np.zeros((n_symbols, self.nfft), dtype=complex)

        # Place data on active carriers
        for i in range(n_symbols):
            ofdm_symbols[i, self.carrier_indices] = symbols[i, :]

        # IFFT to convert to time domain (apply normalization)
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
        symbol_length = self.nfft + self.ncp
        n_symbols = len(received_signal) // symbol_length

        # Truncate signal to complete symbols
        received_signal = received_signal[:n_symbols * symbol_length]

        # Reshape into OFDM symbols
        received_signal = received_signal.reshape(n_symbols, symbol_length)

        # Remove cyclic prefix
        received_signal = received_signal[:, self.ncp:]

        # FFT to convert back to frequency domain (apply normalization)
        freq_signal = np.fft.fft(received_signal, axis=1) / np.sqrt(self.nfft)

        # Extract data from subcarriers
        received_symbols = freq_signal[:, self.carrier_indices]

        return received_symbols.flatten()


class OFDMBeamformer:
    def __init__(self, n_antennas, nfft=64, ncp=16):
        self.n_antennas = n_antennas
        self.nfft = nfft
        self.ncp = ncp
        self.nsym = nfft + ncp  # Total OFDM symbol length

    def calculate_steering_vector(self, angle_degrees):
        """Calculate steering vector for specific angle"""
        angle_rad = np.deg2rad(angle_degrees)
        d = 0.5  # Half wavelength antenna spacing
        k = 2 * np.pi
        n = np.arange(self.n_antennas)
        steering_vector = np.exp(1j * k * d * n * np.sin(angle_rad))

        # Normalize steering vector
        steering_vector = steering_vector / np.sqrt(self.n_antennas)

        return steering_vector

    def apply_beamforming(self, ofdm_signal, angle_degrees):
        """Apply beamforming weights to OFDM symbols"""
        # Calculate number of OFDM symbols
        n_symbols = len(ofdm_signal) // self.nsym

        # Ensure we have complete symbols
        ofdm_signal = ofdm_signal[:n_symbols * self.nsym]

        # Reshape signal into OFDM symbols
        symbols = ofdm_signal.reshape(n_symbols, self.nsym)

        # Calculate steering vector
        steering_vector = self.calculate_steering_vector(angle_degrees)

        # Initialize output array for all antennas
        beamformed = np.zeros((n_symbols, self.nsym, self.n_antennas), dtype=complex)

        # Apply beamforming for each time sample and OFDM symbol
        for i in range(n_symbols):
            for j in range(self.nsym):
                beamformed[i, j, :] = symbols[i, j] * steering_vector

        return beamformed