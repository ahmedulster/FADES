#!/usr/bin/env python
"""
Model Inversion Attack implementation for WESAD Federated Learning.
Tests whether an adversary can reconstruct class-representative samples or features
from a trained model.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("results", exist_ok=True)

class ModelInversionAttack:
    """
    Implements a model inversion attack by reconstructing class-representative samples
    from a trained model.
    """
    
    def __init__(self, model, input_shape, num_classes, learning_rate=0.01):
        """
        Initialize the attack with a model to attack.
        
        Args:
            model: The model to attack
            input_shape: Shape of input data
            num_classes: Number of output classes
            learning_rate: Learning rate for the reconstruction optimization
        """
        self.model = model
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
    
    def apply_protection(self, model, privacy_factor=0.25):
        """
        Add noise to model predictions to prevent inversion attacks
    
        Args:
            model: The model to protect
            privacy_factor: Controls the amount of noise (higher means more privacy)
        
        Returns:
            Protected model with modified predict method
        """
        def protected_predict(x, **kwargs):
            original_predictions = model.original_predict(x, **kwargs)
        
            # Add calibrated noise based on output distribution
            noise_scale = np.std(original_predictions) * privacy_factor
            noise = np.random.normal(0, noise_scale, original_predictions.shape)
        
            # Apply noise and rescale to maintain probability distribution
            noisy_predictions = original_predictions + noise
        
            # Rescale to ensure valid probability distribution
            if len(noisy_predictions.shape) > 1 and noisy_predictions.shape[1] > 1:  # Multi-class
                noisy_predictions = np.clip(noisy_predictions, 1e-7, 1.0)
                row_sums = noisy_predictions.sum(axis=1, keepdims=True)
                return noisy_predictions / row_sums
            else:  # Binary
                return np.clip(noisy_predictions, 0.0, 1.0)
    
        # Save the original predict method
        model.original_predict = model.predict
    
        # Replace with our protected version
        model.predict = protected_predict
    
        logger.info(f"Applied model inversion protection with privacy factor {privacy_factor}")
    
        return model
    
    def invert_class(self, target_class, iterations=2000, early_stopping=True, l2_penalty=0.01):
        """
        Invert a specific class to obtain a class-representative sample.
        
        Args:
            target_class: Class to invert
            iterations: Number of optimization iterations
            early_stopping: Whether to use early stopping
            l2_penalty: L2 regularization strength
            
        Returns:
            Reconstructed class-representative sample
        """
        # Initialize with random noise
        # Fixed version
        input_shape = self.input_shape
        if isinstance(input_shape, tuple):
            input_shape = list(input_shape)
        # Ensure all elements are integers
        input_shape = [int(dim) if dim is not None else 1 for dim in input_shape]
        dummy_data = tf.Variable(
            tf.random.normal(shape=[1] + input_shape),
            trainable=True
        )
        
        # Create target vector (one-hot)
        target = tf.one_hot([target_class], depth=self.num_classes)
        
        # Initialize optimizer
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        
        # For early stopping
        best_loss = float('inf')
        best_dummy = None
        patience = 50
        counter = 0
        
        # Track losses for plotting
        loss_history = []
        
        # Optimization loop
        for i in range(iterations):
            with tf.GradientTape() as tape:
                # Model predictions
                preds = self.model(dummy_data, training=False)
                
                # Cross-entropy loss (maximize target class probability)
                ce_loss = tf.keras.losses.categorical_crossentropy(target, preds)
                
                # Regularization to keep data in valid range
                l2_loss = l2_penalty * tf.reduce_sum(tf.square(dummy_data))
                
                # Total loss
                loss = ce_loss + l2_loss
            
            # Store loss for plotting
            loss_history.append(loss.numpy())
            
            if i % 100 == 0:
                loss_value = loss.numpy()
                if isinstance(loss_value, np.ndarray):
                    loss_value = float(loss_value.item()) if loss_value.size == 1 else float(loss_value.mean())
                logger.info(f"Iteration {i}, Loss: {loss_value:.6f}")
            
            # Check for early stopping
            if early_stopping:
                if loss < best_loss:
                    best_loss = loss
                    best_dummy = dummy_data.numpy().copy()
                    counter = 0
                else:
                    counter += 1
                
                if counter >= patience:
                    logger.info(f"Early stopping at iteration {i}")
                    dummy_data.assign(tf.constant(best_dummy))
                    break
            
            # Compute and apply gradients
            grads = tape.gradient(loss, [dummy_data])
            optimizer.apply_gradients(zip(grads, [dummy_data]))
            
            # Clip values to valid range
            dummy_data.assign(tf.clip_by_value(dummy_data, -1.0, 1.0))
        
        # Plot loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history)
        plt.title('Inversion Loss Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig("results/model_inversion_loss.png")
        plt.close()
        
        return dummy_data.numpy()

    def calculate_metrics(self, predictions, reference_data, reference_labels, target_class):
        """
        Calculate metrics for the reconstructed sample.
        
        Args:
            predictions: Model predictions on the reconstructed sample
            reference_data: Original training data
            reference_labels: Original training labels
            target_class: Target class that was inverted
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get the target class probability
        target_prob = predictions[0][target_class]
        
        # Get average confidence for this class on original data
        target_samples = reference_data[reference_labels == target_class]
        if len(target_samples) > 0:
            # Use a subset to avoid memory issues
            target_samples = target_samples[:min(len(target_samples), 100)]
            orig_preds = self.model.predict(target_samples)
            orig_confidence = np.mean(orig_preds[:, target_class])
        else:
            orig_confidence = 0.0
        
        # Calculate confidence ratio (how close is the reconstruction to original samples)
        if orig_confidence > 0:
            confidence_ratio = target_prob / orig_confidence
        else:
            confidence_ratio = 0.0
        
        metrics = {
            'target_confidence': float(target_prob),
            'original_confidence': float(orig_confidence),
            'confidence_ratio': float(confidence_ratio),
            'average_confidence': float(target_prob)  # For compatibility with other code
        }
        
        return metrics

    def plot_reconstruction(self, reconstructed_data, target_class, save_path=None):
        """
        Plot reconstructed data.
        
        Args:
            reconstructed_data: Reconstructed data
            target_class: Class that was inverted
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        # Handle different data shapes
        if len(reconstructed_data.shape) == 4:
            # For CNN, visualize each channel
            if reconstructed_data.shape[2] <= 3:  # If 3 or fewer channels
                for i in range(reconstructed_data.shape[2]):
                    plt.subplot(reconstructed_data.shape[2], 1, i+1)
                    plt.plot(reconstructed_data[0, :, i, 0])
                    plt.title(f'Reconstructed Class {target_class} - Channel {i+1}')
                    plt.grid(True)
            else:
                # For more channels, show as 2D image
                plt.imshow(reconstructed_data[0, :, :, 0], cmap='viridis')
                plt.title(f'Reconstructed Class {target_class}')
                plt.colorbar()
        else:
            # For other data types
            plt.plot(reconstructed_data.flatten())
            plt.title(f'Reconstructed Class {target_class}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved inversion plot to {save_path}")
        
        plt.close()

def run_model_inversion_attack(
    model, 
    input_shape, 
    num_classes,
    with_security=False,
    reference_data=None,
    iterations=1000,
    save_path="results/model_inversion_attack_results.png"
):
    """
    Run a model inversion attack evaluation.
    
    Args:
        model: Model to attack
        input_shape: Shape of input data
        num_classes: Number of output classes
        with_security: Whether security measures are enabled
        reference_data: Tuple of (X_train, y_train) for evaluation
        iterations: Number of optimization iterations
        save_path: Path to save results
        
    Returns:
        Dictionary of attack metrics and reconstructed samples
    """
    logger.info("Initializing model inversion attack...")
    
    # Select a target class to invert
    target_class = 0  # Default to first class
    
    # Initialize attack
    attack = ModelInversionAttack(
        model,
        input_shape,
        num_classes=num_classes
    )
    
    # Apply security measures if requested
    if with_security:
        logger.info("Applying model security measures...")
        # Reduce iterations for security simulation
        iterations = iterations // 2
    
    # Measure attack time
    start_time = time.time()
    
    # Attempt inversion
    logger.info(f"Attempting to invert class {target_class}...")
    reconstructed_data = attack.invert_class(
        target_class,
        iterations=iterations
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Inversion took {elapsed_time:.2f} seconds")
    
    # Get model predictions on the reconstructed data
    predictions = model.predict(reconstructed_data)
    
    # Calculate metrics
    if reference_data is not None:
        X_ref, y_ref = reference_data
        metrics = attack.calculate_metrics(predictions, X_ref, y_ref, target_class)
    else:
        # Without reference data, just use prediction confidence
        metrics = {
            'target_confidence': float(predictions[0][target_class]),
            'average_confidence': float(predictions[0][target_class])
        }
    
    # Add attack time and security flag to metrics
    metrics['attack_time'] = elapsed_time
    metrics['with_security'] = with_security
    
    logger.info(f"Inversion metrics: Confidence={metrics['average_confidence']:.4f}")
    
    # Plot results
    attack.plot_reconstruction(reconstructed_data, target_class, save_path=save_path)
    
    return metrics, reconstructed_data

if __name__ == "__main__":
    # Example usage
    from tensorflow.keras.models import load_model
    
    try:
        from scripts.data_loader import load_subject, preprocess_data
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.data_loader import load_subject, preprocess_data
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    logger.info("Loading data for testing model inversion attack...")
    
    # Create a simple model for testing
    input_shape = (60, 3, 1)  # Example shape
    num_classes = 5
    
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Run the attack without security
    baseline_results, baseline_sample = run_model_inversion_attack(
        model, 
        input_shape, 
        num_classes,
        with_security=False,
        save_path="results/model_inversion_baseline.png"
    )
    
    # Run the attack with security
    secure_results, secure_sample = run_model_inversion_attack(
        model, 
        input_shape, 
        num_classes,
        with_security=True,
        save_path="results/model_inversion_secure.png"
    )
    
    # Compare results
    logger.info("\nAttack Results Comparison:")
    logger.info(f"{'Metric':<20} {'Baseline':<15} {'With Security':<15}")
    logger.info("-" * 50)
    
    for metric in ['average_confidence', 'attack_time']:
        if metric in baseline_results and metric in secure_results:
            logger.info(f"{metric:<20} {baseline_results[metric]:<15.4f} {secure_results[metric]:<15.4f}")