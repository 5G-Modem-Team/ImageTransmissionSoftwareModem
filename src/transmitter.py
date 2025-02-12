import random
import numpy as np
import matplotlib.pyplot as plt
class Transmitter:
    def __init__(self, bits, modulation = "QPSK"):
        self.bits = bits
        self.modulation = modulation
        self.symbols = None

    def BPSK(self, bits):
        self.modulation = "BPSK"
        self.symbols = bits * 2 - 1
        return self.symbols

    def QPSK(self, bits):
        self.modulation = "QPSK"
        symbols = np.array([(bits[i] * 2 - 1) + 1j*(bits[i+1] * 2 - 1) for i in range(0, len(bits), 2)])
        magnitude = abs(1+1j)
        symbols_normalized = symbols / magnitude # normalize the symbols so have magnitude of 1 for same power as BPSK
        self.symbols = symbols_normalized
        return self.symbols

    def constellation(self, symbols):
        plt.figure()
        plt.plot(np.real(symbols), np.imag(symbols), '.')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.title("Constellation Diagram")
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.grid()
        plt.show()

    def transmit(self):
        if self.modulation == "BPSK":
            symbols = self.BPSK(self.bits)
        elif self.modulation == "QPSK":
            symbols = self.QPSK(self.bits)
        else:
            raise Exception(f"Modulation: {self.modulation} not supported")
        
        return symbols

if __name__ == "__main__":

    bits_string = format(random.getrandbits(100), '0b').zfill(100)
    bit_array = np.array([int(bit) for bit in bits_string])
    tx = Transmitter(bits = bit_array, modulation="QPSK")
    symbols = tx.transmit()
    tx.constellation(symbols)