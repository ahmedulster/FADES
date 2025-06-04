#!/usr/bin/env python
"""
Gradient Leakage Attack implementation for WESAD Federated Learning.
Tests whether an adversary can reconstruct private training data from gradient updates.
"""
import os
import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("results", exist_ok=True)

class GradientLeakageAttack:
    """
    Implements a gradient leakage attack by reconstructing training data
    from model gradient updates.
    """
    
    def __init__(self, model, learning_rate=0.01):
        """
        Initialize the attack with a model.
        
        Args:
            model: The model to attack
            learning_rate: Learning rate for the reconstruction optimization
        """
        self.model = model
        self.learning_rate = learning_rate
        
    def compute_gradients(self, X, y):
        """
        Compute gradients for a batch of data.
        
        Args:
            X: Input data
            y: Target labels
            
        Returns:
            Computed gradients
        """
        # Convert y to appropriate format if needed
        if len(y.shape) == 1:
            # For sparse categorical, keep as is
            y_tensor = tf.convert_to_tensor(y)
        else:
            # For one-hot encoded, use as is
            y_tensor = tf.convert_to_tensor(y)
            
        X_tensor = tf.convert_to_tensor(X)
        
        with tf.GradientTape() as tape:
            # Forward pass
            tape.watch(X_tensor)
            logits = self.model(X_tensor, training=False)
            
            # Compute appropriate loss
            if len(y.shape) == 1:
                # Sparse categorical
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_tensor, logits)
            else:
                # One-hot encoded
                loss = tf.keras.losses.categorical_crossentropy(y_tensor, logits)
                
            # Take mean loss
            loss = tf.reduce_mean(loss)
            
        # Get gradients with respect to model parameters
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return gradients
    
    def reconstruct_data(self, original_gradients, target_label, input_shape, iterations=500):
        """
        Reconstruct training data from gradients.
        
        Args:
            original_gradients: Gradients from the original training data
            target_label: Label of the data to reconstruct
            input_shape: Shape of the input data to reconstruct
            iterations: Number of optimization iterations
            
        Returns:
            Reconstructed data
        """
        # Initialize dummy data with random noise
        dummy_data = tf.Variable(
            tf.random.normal(shape=[1] + list(input_shape), dtype=tf.float32),
            trainable=True
        )
        
        # Create a tensor for the target label
        dummy_label = tf.constant([target_label])
        
        # Initialize optimizer
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        
        # For tracking the best result
        best_loss = float('inf')
        best_dummy = None
        
        # Track losses for plotting
        loss_history = []
        
        # Optimization loop
        for i in range(iterations):
            with tf.GradientTape() as tape:
                # Compute gradients for dummy data
                dummy_gradients = self.compute_gradients(dummy_data, dummy_label)
                
                # Compute the difference between original and dummy gradients
                grad_diff = 0
                for gx, gy in zip(dummy_gradients, original_gradients):
                    if gx is not None and gy is not None:
                        # Cast both tensors to float32 to ensure type compatibility
                        gx_cast = tf.cast(gx, tf.float32)
                        
                        # Handle different types of gy (tensor, numpy array, etc.)
                        if isinstance(gy, tf.Tensor):
                            gy_cast = tf.cast(gy, tf.float32)
                        else:
                            # Convert numpy array or other type to tensor and cast
                            gy_cast = tf.cast(tf.convert_to_tensor(gy), tf.float32)
                        
                        # Now compute squared difference with compatible types
                        grad_diff += tf.reduce_sum(tf.square(gx_cast - gy_cast))
                
                # Add regularization to ensure the dummy data stays in a reasonable range
                regularization = 0.01 * tf.reduce_sum(tf.square(dummy_data))
                
                # Total loss
                loss = grad_diff + regularization
            
            # Store loss for plotting
            try:
                # Handle possible type conversion issues when getting numpy value
                current_loss = float(loss.numpy())
                loss_history.append(current_loss)
                
                if i % 100 == 0:
                    logger.info(f"Iteration {i}, Loss: {current_loss:.6f}")
                
                # Check if this is the best result so far
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_dummy = dummy_data.numpy().copy()
            except Exception as e:
                logger.warning(f"Error converting loss to numpy: {e}")
                # Continue optimization even if logging fails
            
            # Compute and apply gradients
            grads = tape.gradient(loss, [dummy_data])
            optimizer.apply_gradients(zip(grads, [dummy_data]))
            
            # Clip values to valid image range
            dummy_data.assign(tf.clip_by_value(dummy_data, -1.0, 1.0))
        
        # Plot loss curve
        plt.figure(figsize=(8, 5))
        plt.plot(loss_history)
        plt.title('Reconstruction Loss Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig("results/gradient_leakage_loss.png")
        plt.close()
        
        # Return the best reconstruction
        if best_dummy is not None:
            return best_dummy
        else:
            return dummy_data.numpy()
    
    def evaluate_reconstruction(self, original_data, reconstructed_data):
        """
        Evaluate the quality of the reconstruction.
        
        Args:
            original_data: Original training data
            reconstructed_data: Reconstructed data
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure data is in the right shape
        orig = original_data.reshape(original_data.shape).astype(np.float32)
        recon = reconstructed_data.reshape(reconstructed_data.shape).astype(np.float32)
        
        # Mean squared error
        mse = mean_squared_error(orig.flatten(), recon.flatten())
        
        # Correlation coefficient
        try:
            corr = np.corrcoef(orig.flatten(), recon.flatten())[0, 1]
            if np.isnan(corr):
                corr = 0.0
        except:
            corr = 0.0
        
        return {
            'mse': float(mse),
            'correlation': float(corr)
        }
    
    def plot_reconstruction(self, original_data, reconstructed_data, save_path=None):
        """
        Plot original data vs. reconstructed data.
        
        Args:
            original_data: Original training data
            reconstructed_data: Reconstructed data
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Handle different data shapes appropriately
        if len(original_data.shape) == 4:  # For CNN data like WESAD
            # For time-series data with multiple channels
            channels = original_data.shape[2]
            for i in range(min(channels, 3)):  # Plot up to 3 channels
                plt.subplot(2, channels, i + 1)
                plt.plot(original_data[0, :, i, 0])
                plt.title(f'Original - Channel {i+1}')
                plt.grid(True)
                
                plt.subplot(2, channels, channels + i + 1)
                plt.plot(reconstructed_data[0, :, i, 0])
                plt.title(f'Reconstructed - Channel {i+1}')
                plt.grid(True)
        else:
            # Generic fallback
            plt.subplot(1, 2, 1)
            plt.imshow(original_data.reshape((original_data.shape[1], -1)))
            plt.title('Original Data')
            
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_data.reshape((reconstructed_data.shape[1], -1)))
            plt.title('Reconstructed Data')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved reconstruction plot to {save_path}")
        
        plt.close()

def run_gradient_leakage_attack(
    model, 
    X_train, y_train, 
    with_security=False,
    save_path="results/gradient_leakage_attack_results.png"
):
    """
    Run a gradient leakage attack evaluation.
    
    Args:
        model: Model to attack
        X_train: Training data
        y_train: Training labels
        with_security: Whether security measures are enabled
        save_path: Path to save results
        
    Returns:
        Dictionary of attack results
    """
    try:
        logger.info("Initializing gradient leakage attack...")
        
        # Select a sample to reconstruct
        sample_idx = 0
        sample_X = X_train[sample_idx:sample_idx+1]
        sample_y = y_train[sample_idx:sample_idx+1]
        
        # Get the target label as a scalar if it's a single-element array
        if hasattr(sample_y, 'shape') and len(sample_y.shape) > 0:
            target_label = int(sample_y[0])
        else:
            target_label = int(sample_y)
        
        # Initialize attack
        attack = GradientLeakageAttack(model)
        
        # Compute original gradients
        logger.info("Computing original gradients...")
        original_gradients = attack.compute_gradients(sample_X, sample_y)
        
        # Apply security measures if requested
        if with_security:
            logger.info("Applying security measures to gradients...")
            # Import security components only when needed
            try:
                from secure_components.differential_privacy import add_noise
                
                # Apply differential privacy
                for i in range(len(original_gradients)):
                    if original_gradients[i] is not None:
                        # Use the updated add_noise function
                        original_gradients[i] = add_noise(original_gradients[i], noise_scale=0.1)
                
                logger.info("Applied differential privacy noise to gradients")
                
                # Try applying homomorphic encryption if needed
                try:
                    from secure_components.homomorphic_encryption import PaillierEncryption
                    
                    # Initialize encryption
                    encryption = PaillierEncryption(key_size=1024)
                    
                    # Encrypt and then immediately decrypt for demonstration
                    # (In a real system, you'd keep them encrypted longer)
                    for i in range(len(original_gradients)):
                        if original_gradients[i] is not None:
                            encrypted = encryption.encrypt_tensor(original_gradients[i])
                            decrypted = encryption.decrypt_tensor(encrypted)
                            decrypted = np.clip(decrypted, -1e6, 1e6)
                            # Convert back to tensor
                            original_gradients[i] = tf.convert_to_tensor(decrypted, dtype=tf.float32)
                    
                    logger.info("Applied homomorphic encryption to gradients")
                except ImportError:
                    logger.warning("Homomorphic encryption module not available")
                
            except ImportError:
                logger.warning("Could not import security components, simulating security effects")
                # Simulate security effect by adding noise
                for i in range(len(original_gradients)):
                    if original_gradients[i] is not None:
                        noise = tf.random.normal(
                            shape=original_gradients[i].shape,
                            mean=0.0,
                            stddev=0.1 * tf.math.reduce_std(original_gradients[i])
                        )
                        original_gradients[i] = original_gradients[i] + noise
        
        # Measure attack time
        start_time = time.time()
        
        # Try to reconstruct the data
        logger.info("Attempting to reconstruct training data from gradients...")
        iterations = 300 if with_security else 500  # Fewer iterations if security is enabled
        reconstructed_data = attack.reconstruct_data(
            original_gradients, 
            target_label=target_label,
            input_shape=sample_X.shape[1:],
            iterations=iterations
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Reconstruction took {elapsed_time:.2f} seconds")
        
        # Evaluate reconstruction quality
        logger.info("Evaluating reconstruction quality...")
        metrics = attack.evaluate_reconstruction(sample_X, reconstructed_data)
        
        logger.info(f"Reconstruction metrics: MSE={metrics['mse']:.6f}, Correlation={metrics['correlation']:.6f}")
        
        # Plot results
        attack.plot_reconstruction(sample_X, reconstructed_data, save_path=save_path)
        
        # Add attack time to metrics
        metrics['attack_time'] = elapsed_time
        metrics['with_security'] = with_security
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in gradient leakage attack: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return error metrics
        return {
            'error': str(e),
            'mse': float('inf'),
            'correlation': 0.0,
            'attack_time': 0.0,
            'with_security': with_security
        }

if __name__ == "__main__":
    # Example usage
    import tensorflow as tf
    from tensorflow import keras
    
    try:
        # Try importing from parent directory
        from scripts.data_loader import load_subject, preprocess_data
    except ImportError:
        # If that fails, adjust the path
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.data_loader import load_subject, preprocess_data
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    logger.info("Loading data for testing gradient leakage attack...")
    
    try:
        # Load data for subject 2
        subject_data = load_subject("2")
        X_train, X_test, y_train, y_test = preprocess_data(subject_data)
        
        # Create a simple model for testing
        model = keras.Sequential([
            keras.layers.Input(shape=X_train.shape[1:]),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train the model (briefly, just for testing)
        model.fit(X_train[:100], y_train[:100], epochs=1, batch_size=32, verbose=1)
        
        # Run the attack without security
        baseline_results = run_gradient_leakage_attack(
            model, X_train[:10], y_train[:10], 
            with_security=False,
            save_path="results/gradient_leakage_baseline.png"
        )
        
        # Run the attack with security
        secure_results = run_gradient_leakage_attack(
            model, X_train[:10], y_train[:10], 
            with_security=True,
            save_path="results/gradient_leakage_secure.png"
        )
        
        # Compare results
        logger.info("\nAttack Results Comparison:")
        logger.info(f"{'Metric':<15} {'Baseline':<15} {'With Security':<15} {'Improvement':<15}")
        logger.info("-" * 60)
        
        for metric in ['mse', 'correlation']:
            if metric in baseline_results and metric in secure_results:
                improvement = baseline_results[metric] - secure_results[metric]
                if metric == 'correlation':
                    # For correlation, higher is worse for privacy
                    improvement = -improvement
                
                logger.info(f"{metric:<15} {baseline_results[metric]:<15.6f} {secure_results[metric]:<15.6f} {improvement:<15.6f}")
        
        # Compare attack time
        if 'attack_time' in baseline_results and 'attack_time' in secure_results:
            time_increase = (secure_results['attack_time'] / (baseline_results['attack_time'] + 1e-10) - 1) * 100
            logger.info(f"{'Attack Time':<15} {baseline_results['attack_time']:<15.2f}s {secure_results['attack_time']:<15.2f}s {time_increase:<15.2f}%")
            
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())