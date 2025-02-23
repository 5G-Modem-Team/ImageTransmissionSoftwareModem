import numpy as np
from enum import Enum


class ModulationType(Enum):
    BPSK = 1
    QPSK = 2
    QAM16 = 4
    QAM64 = 6


class AdaptiveModulation:
    def __init__(self):
        # SNR thresholds for different modulation schemes
        self.snr_thresholds = {
            ModulationType.BPSK: 4,  # Min SNR for BPSK
            ModulationType.QPSK: 8,  # Min SNR for QPSK
            ModulationType.QAM16: 15,  # Min SNR for 16-QAM
            ModulationType.QAM64: 20  # Min SNR for 64-QAM
        }

    def estimate_snr(self, received_signal, noise_variance):
        """Estimate SNR from received signal"""
        signal_power = np.mean(np.abs(received_signal) ** 2)
        snr_db = 10 * np.log10(signal_power / noise_variance)
        return snr_db

    def select_modulation(self, snr_db):
        """Select appropriate modulation based on SNR"""
        # Handle array input
        if isinstance(snr_db, np.ndarray):
            result = np.full(snr_db.shape, ModulationType.BPSK)
            result[snr_db >= self.snr_thresholds[ModulationType.QPSK]] = ModulationType.QPSK
            result[snr_db >= self.snr_thresholds[ModulationType.QAM16]] = ModulationType.QAM16
            result[snr_db >= self.snr_thresholds[ModulationType.QAM64]] = ModulationType.QAM64
            return result
        else:
            # Handle scalar input
            if snr_db >= self.snr_thresholds[ModulationType.QAM64]:
                return ModulationType.QAM64
            elif snr_db >= self.snr_thresholds[ModulationType.QAM16]:
                return ModulationType.QAM16
            elif snr_db >= self.snr_thresholds[ModulationType.QPSK]:
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

    def optimize_modulation(self, channel_response, target_ber=1e-3):
        """Optimize modulation scheme for each subcarrier"""
        subcarrier_snr = 10 * np.log10(np.abs(channel_response) ** 2)
        # Add margin for target BER
        effective_snr = subcarrier_snr - 3  # 3dB margin for reliability
        return self.select_modulation(effective_snr)

    def adapt_to_channel(self, channel_response, noise_variance):
        """Adapt transmission parameters to channel conditions"""
        # Get average SNR
        avg_snr = self.estimate_snr(channel_response, noise_variance)

        # Get base modulation scheme
        base_mod = self.select_modulation(avg_snr)

        # Calculate achievable data rate
        capacity = self.calculate_channel_capacity(avg_snr)
        actual_rate = self.get_modulation_efficiency(base_mod)

        # Optimize per subcarrier if using OFDM
        if len(channel_response.shape) > 1:  # OFDM case
            subcarrier_mods = self.optimize_modulation(channel_response)
            return {
                'base_modulation': base_mod,
                'subcarrier_modulations': subcarrier_mods,
                'channel_capacity': capacity,
                'actual_rate': actual_rate,
                'average_snr': avg_snr
            }

        return {
            'base_modulation': base_mod,
            'channel_capacity': capacity,
            'actual_rate': actual_rate,
            'average_snr': avg_snr
        }

    def adaptive_modulate(self, data, channel_state):
        """Apply adaptive modulation to data"""
        # Get channel conditions
        adaptation = self.adapt_to_channel(channel_state['H'],
                                         channel_state['noise_variance'])

        # If using OFDM, modulate each subcarrier differently
        if 'subcarrier_modulations' in adaptation:
            modulated_data = []
            data_index = 0
            subcarrier_mods = adaptation['subcarrier_modulations']

            for mod in subcarrier_mods:
                bits_per_symbol = self.get_modulation_efficiency(mod)
                # Take appropriate number of bits for this modulation
                if data_index + bits_per_symbol <= len(data):
                    symbol_bits = data[data_index:data_index + bits_per_symbol]
                else:
                    # Pad with zeros if we run out of data
                    symbol_bits = np.pad(data[data_index:],
                                       (0, bits_per_symbol - (len(data) - data_index)))
                modulated_data.append(self.modulate_symbol(symbol_bits, mod))
                data_index += bits_per_symbol

            return np.array(modulated_data), adaptation

        # Single carrier case
        else:
            mod = adaptation['base_modulation']
            symbols = []
            bits_per_symbol = self.get_modulation_efficiency(mod)

            for i in range(0, len(data), bits_per_symbol):
                symbol_bits = data[i:i + bits_per_symbol]
                if len(symbol_bits) < bits_per_symbol:
                    symbol_bits = np.pad(symbol_bits,
                                       (0, bits_per_symbol - len(symbol_bits)))
                symbols.append(self.modulate_symbol(symbol_bits, mod))

            return np.array(symbols), adaptation

    def modulate_symbol(self, bits, mod_type):
        """Modulate bits according to specified modulation"""
        if mod_type == ModulationType.BPSK:
            return 2 * bits[0] - 1
        elif mod_type == ModulationType.QPSK:
            return ((2 * bits[0] - 1) + 1j * (2 * bits[1] - 1)) / np.sqrt(2)
        elif mod_type == ModulationType.QAM16:
            real_bits = bits[0:2]
            imag_bits = bits[2:4]
            real_val = 2 * real_bits[0] - 1 + (real_bits[1] * 2 - 1)
            imag_val = 2 * imag_bits[0] - 1 + (imag_bits[1] * 2 - 1)
            return (real_val + 1j * imag_val) / np.sqrt(10)
        elif mod_type == ModulationType.QAM64:
            real_bits = bits[0:3]
            imag_bits = bits[3:6]
            real_val = 4 * real_bits[0] + 2 * real_bits[1] + real_bits[2] - 3.5
            imag_val = 4 * imag_bits[0] + 2 * imag_bits[1] + imag_bits[2] - 3.5
            return (real_val + 1j * imag_val) / np.sqrt(42)