import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Tuple, Optional, Union

def add_noise(tensor, noise_scale=0.1):
    """
    Add Gaussian noise to a tensor for differential privacy.
    
    Args:
        tensor: Input tensor
        noise_scale: Standard deviation of the noise
        
    Returns:
        Tensor with added noise
    """
    # Convert numpy array to tensor if needed
    if isinstance(tensor, np.ndarray):
        tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
    
    # Cast tensor to float32 to ensure consistent type
    tensor = tf.cast(tensor, tf.float32)
    
    # Generate noise with the same shape
    noise = tf.random.normal(
        shape=tf.shape(tensor),
        mean=0.0,
        stddev=noise_scale,
        dtype=tf.float32
    )
    
    # Add noise to tensor
    noisy_tensor = tensor + noise
    
    return noisy_tensor

def compute_epsilon(noise_scale: float, delta: float = 1e-5, sensitivity: float = 1.0) -> float:
    """
    Compute epsilon value for given Gaussian noise scale
    
    Args:
        noise_scale: Scale of the noise (sigma)
        delta: Target delta value
        sensitivity: L2 sensitivity of the function
        
    Returns:
        Corresponding epsilon value
    """
    # Based on the analytic Gaussian mechanism
    # Reference: Dwork & Roth, "The Algorithmic Foundations of Differential Privacy"
    
    # This is a simplified approximation; in practice, use a DP library for accurate accounting
    c = np.sqrt(2 * np.log(1.25 / delta))
    epsilon = c * sensitivity / noise_scale
    
    return epsilon

def apply_dp_to_gradients(gradients, noise_scale: float, sensitivity: float = 1.0):
    """
    Apply differential privacy to model gradients
    
    Args:
        gradients: List of gradients (typically from model.trainable_variables)
        noise_scale: Scale of the noise
        sensitivity: L2 sensitivity
        
    Returns:
        List of noisy gradients
    """
    noisy_gradients = []
    for grad in gradients:
        if grad is not None:
            # Use TensorFlow operations directly
            shape = tf.shape(grad)
            dtype = grad.dtype
            noise = tf.random.normal(shape=shape, mean=0.0, 
                                    stddev=noise_scale * sensitivity,
                                    dtype=dtype)
            noisy_gradients.append(grad + noise)
        else:
            noisy_gradients.append(None)
    
    return noisy_gradients

def dp_train_step(model, x_batch, y_batch, optimizer, loss_fn, noise_scale: float = 0.1):
    """
    Perform a training step with differential privacy
    
    Args:
        model: TensorFlow model
        x_batch: Input batch
        y_batch: Target batch
        optimizer: TensorFlow optimizer
        loss_fn: Loss function
        noise_scale: Scale of noise to add
        
    Returns:
        Loss value
    """
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(x_batch, training=True)
        # Calculate loss
        loss = loss_fn(y_batch, predictions)
    
    # Get gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Apply differential privacy
    noisy_gradients = apply_dp_to_gradients(gradients, noise_scale)
    
    # Apply gradients
    optimizer.apply_gradients(zip(noisy_gradients, model.trainable_variables))
    
    return loss

@tf.function
def train_step(x, y, model, noise_scale=0.1):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits))
    
    # Get gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Add noise directly using TensorFlow operations instead of numpy conversion
    noisy_gradients = []
    for grad in gradients:
        if grad is not None:
            # Use TensorFlow noise generation instead of numpy conversion
            noise = tf.random.normal(tf.shape(grad), mean=0.0, stddev=noise_scale)
            noisy_gradients.append(grad + noise)
        else:
            noisy_gradients.append(None)
    
    # Apply gradients
    model.optimizer.apply_gradients(zip(noisy_gradients, model.trainable_variables))
    
    return loss