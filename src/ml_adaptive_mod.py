import numpy as np
from enum import Enum
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ModulationType(Enum):
    BPSK = 1
    QPSK = 2
    QAM16 = 4
    QAM64 = 6


class MLAdaptiveModulation:
    def __init__(self):
        # Conservative SNR thresholds as backup
        self.snr_thresholds = {
            ModulationType.BPSK: 4,  # Min SNR for BPSK
            ModulationType.QPSK: 15,  # SNR for QPSK
            ModulationType.QAM16: 25,  # SNR for 16QAM
            ModulationType.QAM64: 32  # SNR for 64QAM
        }

        # ML components
        self.model = None
        self.scaler = None

        # Initialize by training a model right away
        self._train_model()

    def _extract_enhanced_channel_features(self, channel_matrix):
        """Extract richer channel features for better ML decision making"""
        if channel_matrix is None or np.size(channel_matrix) == 0:
            return np.zeros(12)

        # Calculate basic statistical features
        h_abs = np.abs(channel_matrix)
        h_phase = np.angle(channel_matrix)

        try:
            # Calculate singular values
            s = np.linalg.svd(channel_matrix, compute_uv=False)

            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvals(np.dot(channel_matrix.conj().T, channel_matrix))

            # Calculate condition number (measure of channel matrix stability)
            cond = np.max(s) / np.min(s) if np.min(s) > 1e-10 else 1000

            # Calculate matrix norm
            frobenius_norm = np.sqrt(np.sum(h_abs ** 2))

            # Channel correlation metric
            correlation = np.mean(np.abs(np.corrcoef(h_abs)))

            features = [
                np.mean(h_abs),  # Average gain
                np.std(h_abs),  # Gain variation
                np.max(h_abs),  # Maximum gain
                np.min(h_abs),  # Minimum gain
                np.mean(h_phase),  # Average phase
                np.std(h_phase),  # Phase variation
                cond,  # Condition number
                np.mean(s),  # Average singular value
                np.std(s),  # Singular value variation
                frobenius_norm,  # Frobenius norm
                correlation,  # Channel correlation
                np.std(eigenvalues)  # Eigenvalue distribution
            ]
        except:
            # Use simplified features on error
            features = [
                np.mean(h_abs),
                np.std(h_abs),
                np.max(h_abs),
                np.min(h_abs),
                0, 0, 10, 1, 0, 1, 0.5, 0
            ]

        return np.array(features)

    def _calculate_theoretical_ber(self, snr_db, channel_matrix, channel_type='rayleigh'):
        """Calculate more accurate theoretical BER based on channel conditions"""
        snr_linear = 10 ** (snr_db / 10)

        # Calculate channel quality metrics
        if channel_matrix is not None and np.size(channel_matrix) > 0:
            # Eigenvalue analysis for MIMO channels
            try:
                eigenvalues = np.linalg.eigvals(np.dot(channel_matrix.conj().T, channel_matrix))
                condition_number = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))
                channel_gain = np.mean(np.abs(eigenvalues))
            except:
                condition_number = 1
                channel_gain = 1
        else:
            condition_number = 1
            channel_gain = 1

        # Apply channel effects to SNR
        if channel_type == 'rayleigh':
            effective_snr = snr_linear * channel_gain / (condition_number * 2)
        elif channel_type == 'rician':
            effective_snr = snr_linear * channel_gain / (condition_number * 1.5)
        else:  # AWGN
            effective_snr = snr_linear * channel_gain / condition_number

        # Q-function approximation: Q(x) ≈ 0.5 * exp(-x²/2)
        q_approx = lambda x: 0.5 * np.exp(-x ** 2 / 2)

        # Calculate BER based on modulation
        # BPSK: BER = Q(√(2*SNR))
        bpsk_ber = q_approx(np.sqrt(2 * effective_snr))

        # QPSK: BER = Q(√SNR)
        qpsk_ber = q_approx(np.sqrt(effective_snr))

        # 16-QAM: BER ≈ 3/4 * Q(√(SNR/5))
        qam16_ber = 0.75 * q_approx(np.sqrt(effective_snr / 5))

        # 64-QAM: BER ≈ 7/12 * Q(√(SNR/21))
        qam64_ber = (7 / 12) * q_approx(np.sqrt(effective_snr / 21))

        # Return array of BERs for each modulation
        return np.array([bpsk_ber, qpsk_ber, qam16_ber, qam64_ber])

    def _train_model(self):
        """Train ML model with realistic channel data and better features"""
        print("Training enhanced adaptive modulation model with realistic channels...")

        # Create more realistic training data
        X = []
        y = []
        np.random.seed(42)

        # Simulate real channel scenarios
        channel_scenarios = [
            {'type': 'awgn', 'samples': 500},
            {'type': 'rayleigh', 'samples': 1500, 'doppler': [0.01, 0.05, 0.1]},
            {'type': 'rician', 'samples': 1500, 'k_factor': [1, 3, 5, 10]},
            {'type': 'frequency_selective', 'samples': 1000, 'taps': [2, 3, 5]},
            {'type': 'time_varying', 'samples': 1000, 'coherence': [0.5, 0.8, 0.95]}
        ]

        # BER targets for different data types
        target_ber = {
            'image': 1e-4,  # Images need lower BER
            'text': 1e-3,  # Text can accept slightly higher BER
            'general': 1e-2  # General data BER requirements
        }

        # Generate samples for each channel type
        for scenario in channel_scenarios:
            channel_type = scenario['type']

            for _ in range(scenario['samples']):
                # Random SNR
                snr = np.random.uniform(0, 40)

                # Data size and type
                data_size = np.random.choice([1000, 10000, 50000, 131072, 262144])
                is_image = 1 if data_size >= 50000 else 0
                data_type = 'image' if is_image == 1 else 'general'

                # Generate realistic channel matrix
                if channel_type == 'awgn':
                    h_matrix = np.eye(4) * np.random.uniform(0.9, 1.1)

                elif channel_type == 'rayleigh':
                    doppler = np.random.choice(scenario.get('doppler', [0.01]))
                    h_matrix = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / np.sqrt(2)
                    # Add time-varying effects
                    if doppler > 0.05:
                        h_matrix = h_matrix * (1 + 0.2 * np.random.randn(4, 4))

                elif channel_type == 'rician':
                    k = np.random.choice(scenario.get('k_factor', [3]))
                    los = np.ones((4, 4))
                    nlos = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / np.sqrt(2)
                    h_matrix = np.sqrt(k / (k + 1)) * los + np.sqrt(1 / (k + 1)) * nlos

                elif channel_type == 'frequency_selective':
                    taps = np.random.choice(scenario.get('taps', [3]))
                    h_matrix = np.zeros((4, 4), dtype=complex)
                    for t in range(taps):
                        tap_matrix = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / np.sqrt(2)
                        h_matrix += tap_matrix * np.exp(-t)

                else:  # time_varying
                    coherence = np.random.choice(scenario.get('coherence', [0.8]))
                    base_matrix = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / np.sqrt(2)
                    variation = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / np.sqrt(2)
                    h_matrix = coherence * base_matrix + (1 - coherence) * variation

                # Calculate channel condition number and other advanced features
                try:
                    condition_number = np.linalg.cond(h_matrix)
                    eigenvalues = np.linalg.eigvals(np.dot(h_matrix.conj().T, h_matrix))
                    channel_capacity = np.sum(np.log2(1 + snr * np.abs(eigenvalues)))
                except:
                    condition_number = 100
                    channel_capacity = np.log2(1 + snr)

                # Extract enhanced channel features
                channel_features = self._extract_enhanced_channel_features(h_matrix)

                # Create feature vector
                features = [
                    snr,  # SNR
                    data_size,  # Data size
                    np.mean(np.abs(h_matrix)),  # Average channel gain
                    np.std(np.abs(h_matrix)),  # Channel gain std dev
                    condition_number,  # Channel condition number
                    channel_capacity,  # Theoretical channel capacity
                    np.real(np.linalg.det(h_matrix)),  # Real part of channel matrix determinant
                    np.imag(np.linalg.det(h_matrix)),  # Imaginary part of channel matrix determinant
                    is_image  # Is image data flag
                ]
                X.append(features)

                # For image data, be extra conservative
                if is_image:
                    if channel_type == 'awgn' and snr > 25:
                        mod = 1  # Use QPSK for images only at high SNR
                    else:
                        mod = 0  # Use BPSK otherwise
                else:
                    # Calculate theoretical BER for each modulation
                    bers = self._calculate_theoretical_ber(snr, h_matrix, channel_type)

                    # Select modulation based on BER threshold
                    target = target_ber[data_type]
                    mod = 0  # Default to BPSK
                    for m in range(4):
                        if bers[m] < target:
                            mod = m

                    # Limit to QPSK for large data sizes
                    if data_size > 100000 and mod > 1:
                        mod = 1

                y.append(mod)

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        print(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples")

        # Create ensemble model with multiple algorithms
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        nn = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)

        # Use voting classifier for more robust predictions
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('nn', nn)],
            voting='soft'  # Use probability weighted voting
        )

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Calculate training and validation accuracy
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val)

        print(f"Enhanced model trained - Training accuracy: {train_score:.4f}, Validation accuracy: {val_score:.4f}")

        # Print feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            try:
                importances = self.model.estimators_[0].feature_importances_
            except:
                importances = np.ones(X_train.shape[1]) / X_train.shape[1]

        feature_names = [
            'SNR', 'Data Size', 'Channel Gain', 'Channel Std',
            'Condition Number', 'Channel Capacity', 'Det Real', 'Det Imag', 'Is Image'
        ]

        indices = np.argsort(importances)[::-1]
        print("Feature ranking for modulation selection:")
        for i in range(len(feature_names)):
            if i < len(indices):
                print(f"{i + 1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")

    def estimate_snr(self, received_signal, noise_variance):
        """Estimate SNR using a more robust method"""
        if noise_variance <= 0:
            return 30  # Default high value if noise variance is invalid

        # More robust SNR estimation
        signal_power = np.mean(np.abs(received_signal) ** 2)

        # Add small constant to prevent division by zero
        noise_variance = max(noise_variance, 1e-10)

        # Calculate SNR with clipping to reasonable range
        snr_linear = signal_power / noise_variance
        snr_db = 10 * np.log10(snr_linear)

        return np.clip(snr_db, 0, 40)

    def select_modulation(self, snr_db, data_size=0, channel_matrix=None):
        # Detect if this is likely image data
        is_image = data_size > 50000

        # Apply a more moderate safety margin
        safety_margin = 3 if is_image else 1  # REDUCED from 5/2
        effective_snr = max(0, snr_db - safety_margin)

        # Use ML model if available
        if self.model is not None and self.scaler is not None:
            try:
                # Create feature vector
                feature_vector = np.array([[
                    effective_snr,
                    data_size,
                    np.mean(np.abs(channel_matrix)) if channel_matrix is not None else 1,
                    np.std(np.abs(channel_matrix)) if channel_matrix is not None else 0.1,
                    np.linalg.cond(channel_matrix) if channel_matrix is not None else 10,
                    np.log2(1 + 10 ** (effective_snr / 10)),
                    np.real(np.linalg.det(channel_matrix)) if channel_matrix is not None else 1,
                    np.imag(np.linalg.det(channel_matrix)) if channel_matrix is not None else 0,
                    1 if is_image else 0
                ]])

                # Standardize features
                feature_vector_scaled = self.scaler.transform(feature_vector)

                # Get prediction and confidence
                mod_idx = self.model.predict(feature_vector_scaled)[0]
                mod_probs = self.model.predict_proba(feature_vector_scaled)[0]
                confidence = mod_probs[mod_idx]

                # Map to modulation type
                mod_types = [ModulationType.BPSK, ModulationType.QPSK,
                             ModulationType.QAM16, ModulationType.QAM64]

                selected_mod = mod_types[mod_idx]

                # Lower confidence threshold to 0.5
                if confidence < 0.5 and mod_idx > 0:  # REDUCED from 0.6
                    print(
                        f"Lower confidence ({confidence:.2f}) for {selected_mod.name}, trying {mod_types[mod_idx - 1].name}")
                    selected_mod = mod_types[mod_idx - 1]

                # MODIFIED: Don't cap image data at 16QAM - allow ML to decide
                # Just add a slight penalty for high-order modulations with images
                if is_image and selected_mod == ModulationType.QAM64 and confidence < 0.7:
                    selected_mod = ModulationType.QAM16
                    print(f"Using {selected_mod.name} for image data (confidence below threshold)")

                print(f"ML selected {selected_mod.name} with {confidence:.2f} confidence")
                return selected_mod

            except Exception as e:
                print(f"Error in ML modulation selection: {str(e)}")
                # Fall back to rule-based selection

        # Fall back to conservative rule-based selection
        if is_image:
            return ModulationType.BPSK
        elif effective_snr >= 30:
            return ModulationType.QAM64
        elif effective_snr >= 20:
            return ModulationType.QAM16
        elif effective_snr >= 10:
            return ModulationType.QPSK
        else:
            return ModulationType.BPSK

    def calculate_channel_capacity(self, snr_db, bandwidth=1.0):
        """Calculate Shannon channel capacity with practical margin"""
        practical_snr = max(0, snr_db - 3)  # 3dB implementation margin
        return bandwidth * np.log2(1 + 10 ** (practical_snr / 10))

    def get_modulation_efficiency(self, mod_type):
        """Get bits per symbol for each modulation"""
        if isinstance(mod_type, np.ndarray):
            return np.array([m.value for m in mod_type])
        return mod_type.value

    def estimate_ber(self, snr_db, modulation):
        """
        Estimate BER for given SNR and modulation

        Args:
            snr_db: Signal-to-noise ratio in dB
            modulation: Modulation scheme

        Returns:
            Estimated bit error rate
        """
        snr_linear = 10 ** (snr_db / 10)

        if modulation == ModulationType.BPSK:
            # BPSK BER estimation (Q-function approximation)
            return 0.5 * np.exp(-snr_linear)

        elif modulation == ModulationType.QPSK:
            # QPSK has same BER as BPSK for same Eb/N0
            return 0.5 * np.exp(-snr_linear / 2)

        elif modulation == ModulationType.QAM16:
            # 16-QAM approximate BER (more conservative)
            return 0.75 * np.exp(-snr_linear / 10)

        elif modulation == ModulationType.QAM64:
            # 64-QAM approximate BER (more conservative)
            return 1.5 * np.exp(-snr_linear / 42)

        return 0.5  # Default high BER if unknown modulation

    def adapt_to_channel(self, channel_response, noise_variance, data_size=0):
        """
        Adapt transmission parameters to channel conditions

        Args:
            channel_response: Channel response matrix
            noise_variance: Noise variance
            data_size: Size of data to transmit (bits)

        Returns:
            Dictionary with adaptation parameters
        """
        # Detect if this is likely image data
        is_image = data_size > 50000

        # Get average SNR
        avg_snr = self.estimate_snr(channel_response, noise_variance)

        # Apply different SNR margin based on data type
        effective_snr = avg_snr - (10 if is_image else 5)

        # Select modulation scheme
        base_mod = self.select_modulation(avg_snr, data_size, channel_response)

        # Calculate achievable data rate
        capacity = self.calculate_channel_capacity(avg_snr)
        actual_rate = self.get_modulation_efficiency(base_mod)

        # Estimate BER with adjustment for image data
        estimated_ber = self.estimate_ber(effective_snr, base_mod)
        if is_image:
            # More conservative BER estimate for images
            estimated_ber = min(0.5, estimated_ber * 5)

        return {
            'base_modulation': base_mod,
            'channel_capacity': capacity,
            'actual_rate': actual_rate,
            'average_snr': avg_snr,
            'effective_snr': effective_snr,
            'estimated_ber': estimated_ber,
            'is_image': is_image
        }

    def adapt_with_fec(self, channel_response, noise_variance, data_size=0, enable_ldpc=True):
        """
        Adapt transmission parameters with forward error correction

        Args:
            channel_response: Channel response matrix
            noise_variance: Noise variance
            data_size: Size of data to transmit (bits)
            enable_ldpc: Whether to enable LDPC coding

        Returns:
            Dictionary with modulation and FEC parameters
        """
        # Estimate SNR
        avg_snr = self.estimate_snr(channel_response, noise_variance)

        # Calculate LDPC coding gain (approx 3-6dB)
        coding_gain = 4 if enable_ldpc else 0
        effective_snr = avg_snr + coding_gain

        # Select modulation scheme
        base_mod = self.select_modulation(effective_snr, data_size, channel_response)

        # Use higher code rate for image data
        if data_size > 50000:
            ldpc_rate = 0.8  # 80% information bits
            ldpc_iterations = 20  # More iterations
        else:
            ldpc_rate = 0.5  # 50% information bits
            ldpc_iterations = 10

        return {
            'base_modulation': base_mod,
            'channel_capacity': self.calculate_channel_capacity(avg_snr),
            'effective_snr': effective_snr,
            'original_snr': avg_snr,
            'enable_ldpc': enable_ldpc,
            'ldpc_rate': ldpc_rate,
            'ldpc_iterations': ldpc_iterations,
            'estimated_gain': coding_gain
        }

    def visualize_training_results(self, save_path="results/ml_training_results.png"):
        """Create visualization of ML model training results"""
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import os

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Generate sample predictions for various SNR and data sizes
        snr_values = np.linspace(0, 40, 100)
        data_sizes = [1000, 10000, 50000, 100000, 200000]
        modulations = ["BPSK", "QPSK", "16QAM", "64QAM"]

        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 12))

        # 1. Plot Feature Importance
        plt.subplot(2, 2, 1)
        feature_names = [
            'SNR', 'Data Size', 'Channel Gain', 'Channel Std',
            'Condition Number', 'Channel Capacity', 'Det Real', 'Det Imag', 'Is Image'
        ]

        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            try:
                importances = self.model.estimators_[0].feature_importances_
            except:
                importances = np.ones(len(feature_names)) / len(feature_names)

        indices = np.argsort(importances)
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title('Feature Importance')

        # 2. Plot SNR vs. Modulation for different data sizes
        plt.subplot(2, 2, 2)

        for data_size in [1000, 50000, 200000]:
            mod_choices = []
            for snr in snr_values:
                # Create simple test channel
                test_channel = np.eye(4)
                mod = self.select_modulation(snr, data_size, test_channel)
                mod_choices.append(mod.value)

            plt.plot(snr_values, mod_choices, label=f'Data size: {data_size}')

        plt.xlabel('SNR (dB)')
        plt.ylabel('Bits per Symbol')
        plt.yticks([1, 2, 4, 6], ['BPSK', 'QPSK', '16QAM', '64QAM'])
        plt.title('Modulation Selection vs. SNR')
        plt.grid(True)
        plt.legend()

        # 3. Plot estimated BER vs SNR
        plt.subplot(2, 2, 3)
        for mod in [ModulationType.BPSK, ModulationType.QPSK,
                    ModulationType.QAM16, ModulationType.QAM64]:
            ber_values = []
            for snr in snr_values:
                ber_values.append(self.estimate_ber(snr, mod))
            plt.semilogy(snr_values, ber_values, label=mod.name)

        plt.xlabel('SNR (dB)')
        plt.ylabel('Estimated BER')
        plt.title('Estimated BER vs. SNR')
        plt.grid(True)
        plt.legend()

        # 4. Modulation selection confidence
        plt.subplot(2, 2, 4)

        # Generate data points with various SNRs, channel conditions, and data sizes
        X_vis = []
        labels = []

        for snr in [5, 15, 25, 35]:
            for is_image in [0, 1]:
                for i in range(25):  # 25 random channel conditions
                    channel_cond = np.random.uniform(1, 20)
                    data_size = 50000 if is_image else 1000

                    feature_vector = np.array([[
                        snr,
                        data_size,
                        np.random.uniform(0.5, 1.5),
                        np.random.uniform(0.1, 0.5),
                        channel_cond,
                        np.log2(1 + 10 ** (snr / 10)),
                        np.random.uniform(-1, 1),
                        np.random.uniform(-1, 1),
                        is_image
                    ]])

                    # Transform and predict
                    X_vis.append(feature_vector[0])
                    feature_vector_scaled = self.scaler.transform(feature_vector)
                    mod_idx = self.model.predict(feature_vector_scaled)[0]
                    labels.append(mod_idx)

        # Apply t-SNE to visualize high-dimensional data in 2D
        from sklearn.manifold import TSNE
        X_vis = np.array(X_vis)
        tsne = TSNE(n_components=2, random_state=42)
        X_vis_tsne = tsne.fit_transform(X_vis)

        # Plot t-SNE visualization
        for mod_idx in range(4):
            mask = np.array(labels) == mod_idx
            plt.scatter(X_vis_tsne[mask, 0], X_vis_tsne[mask, 1], label=modulations[mod_idx], alpha=0.7)

        plt.title('t-SNE Visualization of Decision Boundaries')
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return save_path

    def reconstruct_image_from_transmission(received_data, expected_shape=None, use_crc=True, use_ldpc=True):
        """
        Reconstruct image from received data

        Args:
            received_data: Received byte data
            expected_shape: Expected image shape (optional)
            use_crc: Whether to use CRC checking
            use_ldpc: Whether to use LDPC decoding
        """
        # Check if data uses LDPC encoding
        if use_ldpc and len(received_data) >= 4 and received_data[0] == 0xAA and received_data[1] == 0xBB:
            try:
                # Add timeout protection
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("LDPC decoding timed out")

                # Set 30 second timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)

                from pyldpc import make_ldpc, decode, get_message

                # Extract original data size
                original_size = received_data[2] | (received_data[3] << 8)
                encoded_data = received_data[4:]

                # Convert data to bits
                received_bits = np.unpackbits(encoded_data)

                # Calculate LDPC parameters - must match encoding parameters
                n = min(original_size * 2, 1000)  # Limit size
                d_v = 2  # Reduced variable node degree
                d_c = 4  # Reduced check node degree
                H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
                k = G.shape[1]

                # Calculate LDPC blocks
                n_blocks = len(received_bits) // n

                # Reshape to LDPC blocks
                received_blocks = received_bits[:n_blocks * n].reshape(n_blocks, n)

                # LDPC decoding
                decoded_bits = np.zeros((n_blocks, k), dtype=int)
                for i in range(n_blocks):
                    # Assume SNR of 10dB (adjustable)
                    decoded_block = decode(H, received_blocks[i], snr=10)
                    decoded_bits[i] = get_message(G, decoded_block)

                # Flatten and pack back to bytes
                decoded_bits = decoded_bits.flatten()

                # Truncate to original size
                bits_needed = original_size * 8
                if len(decoded_bits) > bits_needed:
                    decoded_bits = decoded_bits[:bits_needed]

                # Pack to bytes
                decoded_data = np.packbits(decoded_bits)

                # Ensure decoded data length is correct
                if len(decoded_data) >= original_size:
                    decoded_data = decoded_data[:original_size]
                else:
                    # Pad if too short
                    decoded_data = np.pad(decoded_data, (0, original_size - len(decoded_data)), 'constant')

                # Cancel timeout
                signal.alarm(0)

                # Continue processing decoded data
                received_data = decoded_data

            except Exception as e:
                # Reset alarm if exception
                try:
                    signal.alarm(0)
                except:
                    pass
                print(f"LDPC decoding failed: {str(e)}")
                # Fall back to not using LDPC

        # CRC check
        if use_crc:
            crc = CRC16()
            valid, received_data = crc.check(received_data)
            if not valid:
                print("CRC check failed, image may be corrupted")

        # Process metadata
        if len(received_data) >= 2:
            width = received_data[0]
            height = received_data[1]
            img_data = received_data[2:]
        else:
            # If not enough metadata, use expected shape
            if expected_shape:
                height, width = expected_shape
                img_data = received_data
            else:
                # Cannot determine image dimensions
                print("Cannot determine image dimensions")
                return None

        # Reshape image data
        try:
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            # If data size doesn't match expected size
            expected_size = width * height
            if len(img_array) < expected_size:
                # Not enough data, pad with zeros
                img_array = np.pad(img_array, (0, expected_size - len(img_array)), 'constant')
            elif len(img_array) > expected_size:
                # Too much data, truncate
                img_array = img_array[:expected_size]

            # Reshape to image
            img_array = img_array.reshape((height, width))

            # Apply median filter to remove noise
            from scipy.ndimage import median_filter
            img_array = median_filter(img_array, size=2)

            # If quantization was used, apply inverse quantization
            levels = 16
            img_array = np.round(img_array * (levels - 1) / 255) * (255 / (levels - 1))

            # Create PIL image
            from PIL import Image
            return Image.fromarray(img_array.astype(np.uint8))

        except Exception as e:
            print(f"Image reconstruction failed: {str(e)}")
            return None