import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, SpatialDropout2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import logging

logger = logging.getLogger(__name__)

def create_model(input_shape, num_classes=5, model_type="cnn2d"):
    """
    Create a CNN model suitable for imbalanced time series classification
    
    Args:
        input_shape: Input shape for the model
        num_classes: Number of output classes (WESAD has 5 classes)
        model_type: Type of model to create ("cnn1d" or "cnn2d")
    
    Returns:
        Compiled Keras model
    """
    logger.info(f"Creating {model_type} model with input shape {input_shape} and {num_classes} classes")
    
    # Initialize regularization to prevent overfitting
    reg = l2(0.001)
    
    # Determine appropriate activation and loss function based on number of classes
    if num_classes == 1:
        # Binary classification case
        final_activation = 'sigmoid'
        loss_function = 'binary_crossentropy'
        metrics = [
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
        ]
    else:
        # Multi-class case
        final_activation = 'softmax'
        loss_function = 'sparse_categorical_crossentropy'
        metrics = [
            'accuracy',
            tf.keras.metrics.SparseCategoricalAccuracy(name='categorical_accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]
    
    if model_type == "cnn1d":
        # For 1D data (samples, timesteps, features)
        # Reshape if needed
        if len(input_shape) == 4:  # (timesteps, features, 1)
            input_shape = (input_shape[0], input_shape[1] * input_shape[2])
            
        model = Sequential([
            # Input layer - specify input shape
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', 
                   kernel_regularizer=reg, input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # Global pooling instead of flattening reduces overfitting
            tf.keras.layers.GlobalAveragePooling1D(),
            
            Dense(128, activation='relu', kernel_regularizer=reg),
            BatchNormalization(),
            Dropout(0.4),  # Stronger dropout to prevent overfitting
            
            # Output layer with proper number of classes and activation
            Dense(num_classes, activation=final_activation)
        ])
    else:
        # For 2D data (samples, timesteps, features, 1)
        # Ensure 2D shape
        if len(input_shape) < 3:
            raise ValueError(f"Expected 3D input shape for CNN2D, got {input_shape}")
            
        model = Sequential([
            # Input layer with regularization
            Conv2D(64, kernel_size=(3, 1), activation='relu', padding='same',
                   kernel_regularizer=reg, input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 1)),
            
            Conv2D(128, kernel_size=(3, 1), activation='relu', padding='same', kernel_regularizer=reg),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 1)),
            SpatialDropout2D(0.2),  # Spatial dropout works better for 2D
            
            Conv2D(256, kernel_size=(3, 1), activation='relu', padding='same', kernel_regularizer=reg),
            BatchNormalization(),
            GlobalAveragePooling2D(),  # Better than Flatten for preventing overfitting
            
            Dense(128, activation='relu', kernel_regularizer=reg),
            BatchNormalization(),
            Dropout(0.4),
            
            # Output layer with proper activation
            Dense(num_classes, activation=final_activation)
        ])
    
    # Compile model with appropriate loss function and metrics
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss_function,
        metrics=metrics
    )
    
    model.summary()
    return model


# Alternative model with focal loss for severe class imbalance
def create_focal_loss_model(input_shape, num_classes=5, alpha=0.25, gamma=2.0):
    """
    Create a model with focal loss to better handle class imbalance
    
    Args:
        input_shape: Input shape for the model
        num_classes: Number of output classes
        alpha: Weighting factor for focal loss
        gamma: Focusing parameter for focal loss
    
    Returns:
        Compiled Keras model with focal loss
    """
    # Define focal loss function, adjusted for binary classification if needed
    def focal_loss(y_true, y_pred, gamma=gamma, alpha=alpha):
        # Handle binary classification case
        if num_classes == 1:
            # Binary focal loss
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            # Reshape y_true to match y_pred shape
            y_true = tf.cast(y_true, dtype=tf.float32)
            
            # Binary cross entropy
            bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
            
            # Apply focal loss formula for binary case
            p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
            focal_weight = tf.pow(1 - p_t, gamma)
            
            # Apply alpha if specified
            if alpha is not None:
                alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
                focal_weight = alpha_weight * focal_weight
                
            return tf.reduce_mean(focal_weight * bce)
        else:
            # Multi-class focal loss (original implementation)
            # Convert sparse targets to one-hot
            y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
            
            # Calculate focal loss
            epsilon = 1e-7
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Cross entropy
            cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
            
            # Focal term
            weight = alpha * y_true_one_hot * tf.pow(1 - y_pred, gamma)
            
            # Compute the focal loss
            focal = weight * cross_entropy
            focal = tf.reduce_sum(focal, axis=-1)
            
            return focal
    
    # Create model architecture with appropriate final activation
    model = create_model(input_shape, num_classes, model_type="cnn2d")
    
    # Determine appropriate metrics based on number of classes
    if num_classes == 1:
        metrics = [
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
        ]
    else:
        metrics = [
            'accuracy',
            tf.keras.metrics.SparseCategoricalAccuracy(name='categorical_accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]
    
    # Re-compile with focal loss
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=focal_loss,
        metrics=metrics
    )
    
    return model