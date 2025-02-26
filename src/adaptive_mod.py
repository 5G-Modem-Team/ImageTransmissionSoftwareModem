import numpy as np
from enum import Enum


class ModulationType(Enum):
    BPSK = 1
    QPSK = 2
    QAM16 = 4
    QAM64 = 6


class AdaptiveModulation:
    def __init__(self):
        # Much more conservative SNR thresholds for different modulation schemes
        self.snr_thresholds = {
            ModulationType.BPSK: 4,  # Min SNR for BPSK
            ModulationType.QPSK: 12,  # Increased from 10 to 12
            ModulationType.QAM16: 20,  # Increased from 16 to 20
            ModulationType.QAM64: 27  # Increased from 22 to 27
        }

    def estimate_snr(self, received_signal, noise_variance):
        """Estimate SNR from received signal"""
        signal_power = np.mean(np.abs(received_signal) ** 2)
        snr_db = 10 * np.log10(signal_power / noise_variance)
        return snr_db

    def select_modulation(self, snr_db, data_size=0):
        """
        Select appropriate modulation based on SNR

        Args:
            snr_db: Signal-to-noise ratio in dB
            data_size: Size of data to transmit (optional)

        Returns:
            ModulationType: Selected modulation scheme
        """
        # For image transmission, add 3dB safety margin
        effective_snr = snr_db - 3

        # Handle array input
        if isinstance(effective_snr, np.ndarray):
            result = np.full(effective_snr.shape, ModulationType.BPSK)
            result[effective_snr >= self.snr_thresholds[ModulationType.QPSK]] = ModulationType.QPSK
            result[effective_snr >= self.snr_thresholds[ModulationType.QAM16]] = ModulationType.QAM16
            result[effective_snr >= self.snr_thresholds[ModulationType.QAM64]] = ModulationType.QAM64
            return result
        else:
            # Handle scalar input - default to QPSK for small packets
            if data_size > 0 and data_size < 10000:
                return ModulationType.QPSK

            # For larger data or when size is not specified
            if effective_snr >= self.snr_thresholds[ModulationType.QAM64]:
                return ModulationType.QAM64
            elif effective_snr >= self.snr_thresholds[ModulationType.QAM16]:
                return ModulationType.QAM16
            elif effective_snr >= self.snr_thresholds[ModulationType.QPSK]:
                return ModulationType.QPSK
            else:
                return ModulationType.BPSK

    def calculate_channel_capacity(self, snr_db, bandwidth=1.0):
        """Calculate Shannon channel capacity"""
        return bandwidth * np.log2(1 + 10 ** (snr_db / 10))

    def get_modulation_efficiency(self, mod_type):
        """Get bits per symbol for each modulation"""
        if isinstance(mod_type, np.ndarray):
            return np.array([m.value for m in mod_type])
        return mod_type.value

    def adapt_to_channel(self, channel_response, noise_variance, data_size=0):
        """
        Adapt transmission parameters to channel conditions

        Args:
            channel_response: Channel response matrix
            noise_variance: Noise variance
            data_size: Size of data to transmit (optional)

        Returns:
            Dictionary with adaptation parameters
        """
        # Get average SNR
        avg_snr = self.estimate_snr(channel_response, noise_variance)

        # Get base modulation scheme
        base_mod = self.select_modulation(avg_snr, data_size)

        # For image data, always use QPSK or BPSK for small data
        if data_size > 0 and data_size < 10000:
            if avg_snr < 15:
                base_mod = ModulationType.BPSK
            else:
                base_mod = ModulationType.QPSK

        # Calculate achievable data rate
        capacity = self.calculate_channel_capacity(avg_snr)
        actual_rate = self.get_modulation_efficiency(base_mod)

        return {
            'base_modulation': base_mod,
            'channel_capacity': capacity,
            'actual_rate': actual_rate,
            'average_snr': avg_snr
        }