import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from src.transmitter import Transmitter
from src.receiver import Receiver


def test_modulation_scheme(modulation, num_bits=10000, snr_range=[0, 5, 10, 15, 20, 25, 30]):
    """Test a modulation scheme across different SNR values"""
    # Adjust bit count to match modulation scheme
    if modulation == "QPSK":
        bits = np.random.randint(0, 2, num_bits - (num_bits % 2))
    elif modulation == "16QAM":
        bits = np.random.randint(0, 2, num_bits - (num_bits % 4))
    elif modulation == "64QAM":
        bits = np.random.randint(0, 2, num_bits - (num_bits % 6))
    else:  # BPSK
        bits = np.random.randint(0, 2, num_bits)

    ber_results = []

    for snr_db in snr_range:
        # Initialize components
        tx = Transmitter(bits, modulation=modulation)
        rx = Receiver(modulation=modulation)

        # Modulate (no beamforming)
        tx_symbols = tx.transmit(beam_angle=None)

        # Simple AWGN channel
        signal_power = np.mean(np.abs(tx_symbols) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*tx_symbols.shape) +
                                            1j * np.random.randn(*tx_symbols.shape))
        rx_symbols = tx_symbols + noise

        # Plot constellation for one specific SNR (e.g., 15dB)
        if snr_db == 15:
            plt.figure(figsize=(8, 8))
            plt.scatter(np.real(rx_symbols[:100]), np.imag(rx_symbols[:100]), alpha=0.7)
            plt.grid(True)
            plt.title(f"Received {modulation} Constellation at {snr_db}dB SNR")
            plt.xlabel("Real")
            plt.ylabel("Imaginary")
            plt.axis('equal')
            plt.savefig(f"{modulation}_constellation_{snr_db}dB.png")
            plt.close()  # Close the figure to free memory

        # Demodulate
        rx_bits, _ = rx.receive(rx_symbols)

        # Calculate BER
        min_len = min(len(bits), len(rx_bits))
        ber = np.mean(bits[:min_len] != rx_bits[:min_len])
        ber_results.append(ber)
        print(f"{modulation} at {snr_db}dB SNR: BER = {ber:.6f}")

    return snr_range, ber_results


# Test all modulation schemes
modulations = ["BPSK", "QPSK", "16QAM", "64QAM"]
plt.figure(figsize=(10, 6))

for mod in modulations:
    snr_range, ber_results = test_modulation_scheme(mod)
    plt.semilogy(snr_range, ber_results, marker='o', label=mod)

plt.grid(True)
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs SNR for Different Modulation Schemes")
plt.legend()
plt.savefig("ber_vs_snr.png")
plt.close()  # Close the figure to prevent hanging

print("All tests completed. Check the generated .png files for plots.")