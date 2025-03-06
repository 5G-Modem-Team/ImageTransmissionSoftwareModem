# Image Transmission Software Modem  

## How to Run Tests
### 1. Basic Modem Testing
This script tests individual modulation schemes (BPSK, QPSK, 16QAM, 64QAM) under different SNR levels.
```bash
python test_basic_modem.py
```
- Outputs BER values for each modulation scheme.
- Helps analyze the impact of SNR on different modulation types.

### 2. Full System Testing
This script runs a full pipeline including modulation, OFDM, beamforming, channel effects, and demodulation.
```bash
python test_system.py
```
- Tests multiple images with different complexity levels.
- Evaluates system-wide BER, PSNR, and SSIM metrics.
- Simulates realistic wireless transmission scenarios.


## 🚀 Let's Go Team! 💪  
