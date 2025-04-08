import numpy as np
from scipy import interpolate


class ImprovedOFDMProcessor:
    """Enhanced OFDM processor with synchronization and channel estimation"""

    def __init__(self, nfft=64, ncp=16, ncarriers=52):
        self.nfft = nfft  # FFT size
        self.ncp = ncp  # Cyclic prefix length
        self.ncarriers = ncarriers  # Number of data carriers
        self.nsym = self.nfft + self.ncp  # Total symbol length

        # Generate carrier indices
        self.carrier_indices = np.concatenate([
            np.arange(-ncarriers // 2, 0),
            np.arange(1, ncarriers // 2 + 1)
        ]) % nfft

        # Add synchronization sequence
        np.random.seed(42)  # Use fixed seed for reproducibility
        self.sync_seq = (np.random.randn(self.nfft) + 1j * np.random.randn(self.nfft)) / np.sqrt(2)

        # Add pilot positions (for channel estimation)
        self.pilot_indices = np.arange(0, self.ncarriers, 4)  # Every 4th subcarrier is a pilot
        self.pilot_values = np.ones(len(self.pilot_indices))  # Pilot values are 1

    def modulate(self, symbols):
        """
        Enhanced OFDM modulation with synchronization sequence and pilots

        Args:
            symbols: Input symbols to modulate

        Returns:
            OFDM modulated time domain signal
        """
        # Calculate required number of OFDM symbols
        n_symbols = (len(symbols) + self.ncarriers - 1) // self.ncarriers

        # Pad the input symbols if necessary
        padded_length = n_symbols * self.ncarriers
        if len(symbols) < padded_length:
            symbols = np.pad(symbols, (0, padded_length - len(symbols)), 'constant')

        # Reshape input for OFDM symbols
        symbols = symbols.reshape(n_symbols, self.ncarriers)

        # Prepare OFDM symbols in frequency domain
        ofdm_symbols = np.zeros((n_symbols, self.nfft), dtype=complex)

        # Place data and pilots
        for i in range(n_symbols):
            # Place data on active carriers
            ofdm_symbols[i, self.carrier_indices] = symbols[i, :]

            # Insert pilots in each OFDM symbol
            pilot_data_indices = np.mod(self.pilot_indices, len(self.carrier_indices))
            for j, pilot_idx in enumerate(pilot_data_indices):
                carrier_idx = self.carrier_indices[pilot_idx]
                ofdm_symbols[i, carrier_idx] = self.pilot_values[j]

        # Add synchronization sequence as first symbol
        sync_symbol = np.copy(self.sync_seq)

        # IFFT to convert to time domain
        time_signal = np.fft.ifft(ofdm_symbols, axis=1) * np.sqrt(self.nfft)

        # Add cyclic prefix
        cp = time_signal[:, -self.ncp:]
        ofdm_signal = np.hstack([cp, time_signal])

        # Add synchronization sequence
        sync_time = np.fft.ifft(sync_symbol) * np.sqrt(self.nfft)
        sync_cp = sync_time[-self.ncp:]
        sync_ofdm = np.concatenate([sync_cp, sync_time])

        # Combine synchronization sequence and OFDM signal
        full_signal = np.concatenate([sync_ofdm, ofdm_signal.flatten()])

        return full_signal

    def demodulate(self, received_signal):
        """
        Enhanced OFDM demodulation with synchronization and channel estimation

        Args:
            received_signal: Received time domain signal

        Returns:
            Demodulated symbols
        """
        # First perform synchronization detection
        sync_length = self.nfft + self.ncp

        # If signal is too short, cannot process
        if len(received_signal) < 2 * sync_length:
            print("Warning: Signal too short for synchronization")
            # Try simple demodulation
            return self._simple_demodulate(received_signal)

        # Extract synchronization sequence
        rx_sync = received_signal[:sync_length]
        rx_signal = received_signal[sync_length:]

        # Calculate FFT window start position
        correlation = []
        for i in range(self.ncp):
            corr = np.abs(np.vdot(rx_sync[i:i + self.nfft], np.fft.ifft(self.sync_seq) * np.sqrt(self.nfft)))
            correlation.append(corr)

        # Find maximum correlation position
        sync_start = np.argmax(correlation)

        # Adjust window position
        rx_signal = rx_signal[sync_start:]

        # Calculate number of OFDM symbols
        symbol_length = self.nfft + self.ncp
        n_symbols = len(rx_signal) // symbol_length

        # Truncate to complete symbols
        rx_signal = rx_signal[:n_symbols * symbol_length]

        # Reshape to OFDM symbols
        rx_signal = rx_signal.reshape(n_symbols, symbol_length)

        # Remove cyclic prefix
        rx_signal = rx_signal[:, self.ncp:]

        # FFT to convert to frequency domain
        freq_signal = np.fft.fft(rx_signal, axis=1) / np.sqrt(self.nfft)

        # Channel estimation
        channel_response = np.ones((n_symbols, self.nfft), dtype=complex)

        # Use pilots for interpolated channel estimation
        for i in range(n_symbols):
            try:
                pilot_indices = self.carrier_indices[np.mod(self.pilot_indices, len(self.carrier_indices))]
                pilot_received = freq_signal[i, pilot_indices]

                # Simple linear interpolation

                # Calculate channel response at each pilot
                pilot_response = pilot_received / self.pilot_values

                # Create interpolation function
                f_real = interpolate.interp1d(pilot_indices, np.real(pilot_response),
                                              kind='linear', bounds_error=False, fill_value="extrapolate")
                f_imag = interpolate.interp1d(pilot_indices, np.imag(pilot_response),
                                              kind='linear', bounds_error=False, fill_value="extrapolate")

                # Interpolate all carriers
                for carrier in self.carrier_indices:
                    channel_response[i, carrier] = complex(f_real(carrier), f_imag(carrier))
            except Exception as e:
                # Fall back to no channel estimation
                print(f"Channel estimation failed: {str(e)}, using unit response")

        # Channel equalization
        equalized_signal = np.zeros_like(freq_signal)
        for i in range(n_symbols):
            for carrier in self.carrier_indices:
                if np.abs(channel_response[i, carrier]) > 1e-6:
                    equalized_signal[i, carrier] = freq_signal[i, carrier] / channel_response[i, carrier]
                else:
                    equalized_signal[i, carrier] = freq_signal[i, carrier]

        # Extract data symbols
        received_symbols = equalized_signal[:, self.carrier_indices]

        return received_symbols.flatten()

    def _simple_demodulate(self, received_signal):
        """Simple OFDM demodulation without synchronization"""
        # Calculate number of complete OFDM symbols
        symbol_length = self.nfft + self.ncp
        n_symbols = len(received_signal) // symbol_length

        # Truncate signal to complete symbols
        received_signal = received_signal[:n_symbols * symbol_length]

        # Reshape into OFDM symbols
        received_signal = received_signal.reshape(n_symbols, symbol_length)

        # Remove cyclic prefix
        received_signal = received_signal[:, self.ncp:]

        # FFT to convert back to frequency domain
        freq_signal = np.fft.fft(received_signal, axis=1) / np.sqrt(self.nfft)

        # Extract data from subcarriers
        received_symbols = freq_signal[:, self.carrier_indices]

        return received_symbols.flatten()