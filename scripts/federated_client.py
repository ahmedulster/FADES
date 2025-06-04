#!/usr/bin/env python
"""
Federated learning client implementation for the WESAD dataset with security features.
This client connects to the server and participates in the federated learning process.
"""
import time
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import flwr as fl
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import traceback
import secrets
import matplotlib.pyplot as plt
import seaborn as sns
import base64


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Constants
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import the security components
try:
    from secure_components.homomorphic_encryption import PaillierEncryption
    from secure_components.differential_privacy import add_noise
    from secure_components.zkp import generate_proof, verify_proof
except ImportError as e:
    logger.error(f"Error importing security components: {e}")
    logger.error("Make sure the secure_components module is in your Python path")
    sys.exit(1)

# Import the data loader and model
try:
    from scripts.data_loader import load_subject, preprocess_data
    from model.cnn_model import create_model
except ImportError as e:
    logger.error(f"Error importing data loader or model: {e}")
    logger.error("Make sure the scripts and model modules are in your Python path")
    sys.exit(1)


def get_client_params():
    """
    Parse command line arguments or retrieve from environment variables.
    Environment variables take precedence over command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    # Check for environment variables first
    env_subject_id = os.environ.get("FL_SUBJECT_ID")
    env_window_size = os.environ.get("FL_WINDOW_SIZE")
    env_subset_size = os.environ.get("FL_SUBSET_SIZE")
    env_server_address = os.environ.get("FL_SERVER_ADDRESS")
    env_use_he = os.environ.get("FL_USE_HE")
    env_use_dp = os.environ.get("FL_USE_DP")
    env_dp_epsilon = os.environ.get("FL_DP_EPSILON")
    env_noise_multiplier = os.environ.get("FL_NOISE_MULTIPLIER")
    env_privacy_budget = os.environ.get("FL_PRIVACY_BUDGET")
    env_use_zkp = os.environ.get("FL_USE_ZKP")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Federated Learning Client for WESAD Dataset")
    
    # Basic parameters
    parser.add_argument("--subject_id", type=str, default=env_subject_id or "2",
                        help="Subject ID (e.g., 2, without S prefix)")
    parser.add_argument("--window_size", type=int, 
                        default=int(env_window_size) if env_window_size else 60,
                        help="Window size for time-series data")
    parser.add_argument("--subset_size", type=float, 
                        default=float(env_subset_size) if env_subset_size else 0.2,
                        help="Fraction of data to use (0.0-1.0)")
    parser.add_argument("--server_address", type=str, 
                        default=env_server_address or "localhost:8080",
                        help="Server address (host:port)")
    
    # Security parameters
    parser.add_argument("--use_homomorphic_encryption", action="store_true", 
                        default=env_use_he == "1",
                        help="Enable homomorphic encryption")
    parser.add_argument("--use_differential_privacy", action="store_true", 
                        default=env_use_dp == "1",
                        help="Enable differential privacy")
    parser.add_argument("--use_zero_knowledge_proofs", action="store_true", 
                        default=env_use_zkp == "1",
                        help="Enable zero-knowledge proofs")
    parser.add_argument("--dp_epsilon", type=float, 
                        default=float(env_dp_epsilon) if env_dp_epsilon else 1.0,
                        help="Differential privacy epsilon parameter")
    parser.add_argument("--noise_multiplier", type=float, 
                        default=float(env_noise_multiplier) if env_noise_multiplier else 1.1,
                        help="Noise multiplier for DP")
    parser.add_argument("--privacy_budget", type=float, 
                        default=float(env_privacy_budget) if env_privacy_budget else 5.0,
                        help="Total privacy budget")
    
    args = parser.parse_args()
    return args


def load_and_preprocess_data(subject_id, window_size=60, subset_size=0.2):
    """
    Load and preprocess data for a specific subject.
    
    Args:
        subject_id: Subject ID
        window_size: Window size for time-series data
        subset_size: Fraction of data to use
        
    Returns:
        Tuple of (X_train, y_train), (X_test, y_test)
    """
    try:
        # Clean up subject ID (remove 'S' prefix if present)
        if isinstance(subject_id, str) and subject_id.startswith('S'):
            subject_id = subject_id.replace('S', '')
            
        logger.info(f"Initializing client for subject {subject_id}")
        logger.info(f"Attempting to load data for subject {subject_id}")
        
        # Load subject data
        subject_data = load_subject(subject_id)
        
        # Log data shapes
        logger.info(f"Loaded data shapes - ACC: {subject_data['acc'].shape}, labels: {subject_data['label'].shape}")
        
        # Check original label distribution
        unique_labels, counts = np.unique(subject_data['label'], return_counts=True)
        original_dist = dict(zip(unique_labels, counts))
        logger.info(f"Original label distribution: {original_dist}")
        
        # Apply subset sampling if needed
        if subset_size < 1.0:
            total_samples = len(subject_data['acc'])
            sampled_indices = []
            
            for label in unique_labels:
                label_indices = np.where(subject_data['label'] == label)[0]
                
                # Ensure minimum samples per class
                min_samples = max(200, int(0.1 * len(label_indices)))
                label_sample_size = max(min_samples, int(len(label_indices) * subset_size))
                
                # Cap at actual available samples
                label_sample_size = min(label_sample_size, len(label_indices))
                
                if label_sample_size > 0:
                    # Use same seed for reproducibility but unique per subject and class
                    np.random.seed(42 + int(subject_id) + int(label))
                    sampled_from_class = np.random.choice(
                        label_indices,
                        size=label_sample_size,
                        replace=False
                    )
                    sampled_indices.extend(sampled_from_class)
                    logger.info(f"Sampled {label_sample_size} examples from class {label}")
            
            if sampled_indices:
                sampled_indices = np.sort(np.array(sampled_indices))
                subject_data['acc'] = subject_data['acc'][sampled_indices]
                subject_data['label'] = subject_data['label'][sampled_indices]
                
                # Check sampled distribution
                unique_labels, counts = np.unique(subject_data['label'], return_counts=True)
                sampled_dist = dict(zip(unique_labels, counts))
                logger.info(f"After sampling, label distribution: {sampled_dist}")
                logger.info(f"Using {len(sampled_indices)} samples out of {total_samples}")
        
        # Preprocess data
        logger.info("Starting preprocessing...")
        X_train, X_test, y_train, y_test = preprocess_data(
            subject_data,
            use_temporal_split=True,
            balance=True,
            use_chest=False
        )
        
        # Verify class distribution
        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)
        
        logger.info(f"Training set has classes: {train_classes}")
        logger.info(f"Test set has classes: {test_classes}")
        
        if len(train_classes) < 2 or len(test_classes) < 2:
            logger.warning(f"Insufficient class diversity for subject {subject_id}!")
            
        logger.info(f"Preprocessing complete. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        return (X_train, y_train), (X_test, y_test)
        
    except Exception as e:
        logger.error(f"Error loading or preprocessing data: {e}")
        logger.error(traceback.format_exc())
        raise


class SecureWESADClient(fl.client.NumPyClient):
    """Secure Federated Learning client for WESAD dataset."""
    
    def __init__(
        self,
        subject_id,
        window_size=60,
        subset_size=0.2,
        server_address="localhost:8080",
        use_homomorphic_encryption=False,
        use_differential_privacy=False,
        use_zero_knowledge_proofs=False,
        dp_epsilon=1.0,
        noise_multiplier=1.1,
        privacy_budget=5.0
    ):
        """
        Initialize the client with security features.
        
        Args:
            subject_id: Subject ID
            window_size: Window size for time-series data
            subset_size: Fraction of data to use
            server_address: Server address
            use_homomorphic_encryption: Enable homomorphic encryption
            use_differential_privacy: Enable differential privacy
            use_zero_knowledge_proofs: Enable zero-knowledge proofs
            dp_epsilon: Differential privacy epsilon parameter
            noise_multiplier: Noise multiplier for DP
            privacy_budget: Total privacy budget
        """
        self.subject_id = subject_id
        self.window_size = window_size
        self.subset_size = subset_size
        self.server_address = server_address
        
        # Security settings
        self.use_homomorphic_encryption = use_homomorphic_encryption
        self.use_differential_privacy = use_differential_privacy
        self.use_zero_knowledge_proofs = use_zero_knowledge_proofs
        self.dp_epsilon = dp_epsilon
        self.noise_multiplier = noise_multiplier
        self.privacy_budget = privacy_budget
        self.remaining_privacy_budget = privacy_budget
        
        # Client identity
        self.client_id = f"client_{subject_id}"
        self.secret_key = secrets.token_hex(32)  # For ZKP
        
        # Initialize security components
        if self.use_homomorphic_encryption:
            try:
                self.encryption = PaillierEncryption(key_size=1024)
                logger.info("Generated Paillier keys")
                logger.info("Initialized Paillier encryption with 1024-bit keys")
                logger.info(f"Initialized Paillier encryption with key size {1024}")
            except Exception as e:
                logger.error(f"Failed to initialize homomorphic encryption: {e}")
                self.use_homomorphic_encryption = False
        
        # Load and preprocess data
        try:
            (self.X_train, self.y_train), (self.X_test, self.y_test) = load_and_preprocess_data(
                subject_id, window_size, subset_size
            )
            logger.info(f"Loaded data for subject {subject_id}: X_train shape {self.X_train.shape}, y_train shape {self.y_train.shape}")
            
            # In WESAD, we need to ensure consistent model output dimensions
            self.num_classes = 5  # Standard for WESAD
            
            # Check actual classes present
            unique_classes = np.unique(np.concatenate([self.y_train, self.y_test]))
            logger.info(f"Found {len(unique_classes)} unique classes locally: {unique_classes}")
            logger.info(f"Using model with {self.num_classes} output classes for compatibility")
            
            # Initialize model
            logger.info(f"Creating cnn2d model with input shape {self.X_train.shape[1:]} and {self.num_classes} classes")
            self.model = create_model(
                input_shape=self.X_train.shape[1:],
                num_classes=self.num_classes
                
            )
            # Apply model inversion protection if differential privacy is enabled
            if self.use_differential_privacy:
                try:
        # Import the ModelInversionAttack class
                    from attacks.model_inversion import ModelInversionAttack
        
        # Create a dummy instance just to use the protection method
                    protection = ModelInversionAttack(
                        model=None,  # Not using for attack, just for protection
                        input_shape=self.X_train.shape[1:],
                        num_classes=self.num_classes
                    )
        
        # Apply protection to the model
                    self.model = protection.apply_protection(self.model, privacy_factor=0.3)
                    logger.info("Applied model inversion protection")
                except Exception as e:
                    logger.warning(f"Failed to apply model inversion protection: {str(e)}")
            
            # Initialize training history
            self.history = {
                "train_loss": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_accuracy": []
            }
            
            # Compute noise scale for differential privacy
            if self.use_differential_privacy:
                self.compute_noise_scale()
                
        except Exception as e:
            logger.error(f"Error initializing client: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def compute_noise_scale(self):
        """Compute noise scale for differential privacy."""
        if not self.use_differential_privacy:
            return
            
        # Simple calculation based on epsilon and sensitivity
        sensitivity = 1.0  # Assume L2 sensitivity of 1 for normalized gradients
        self.dp_noise_scale = self.noise_multiplier * sensitivity / self.dp_epsilon
        logger.info(f"Computed DP noise scale: {self.dp_noise_scale}")
    
    def get_parameters(self, config):
        """Return model parameters."""
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        """Train the model on local data."""
        # Update model with server parameters
        self.model.set_weights(parameters)
        
        # Check privacy budget
        if self.use_differential_privacy and self.remaining_privacy_budget <= 0:
            logger.warning("Privacy budget exhausted, skipping training")
            return parameters, 0, {}
            
        # Handle ZKP challenge if present and ZKP is enabled
        proof = {}
        if self.use_zero_knowledge_proofs:
            if "zkp_challenge" in config and "client_id" in config:
                challenge = config.get("zkp_challenge")
                server_client_id = config.get("client_id")
                
                # Generate proof based on challenge
                try:
                    proof = generate_proof(self.secret_key, challenge, server_client_id)
                    self.client_id = server_client_id
                    logger.info(f"Generated ZKP proof for challenge from server")
                except Exception as e:
                    logger.error(f"Failed to generate ZKP proof: {str(e)}")
                    proof = {}
            else:
                logger.warning("No ZKP challenge received from server despite ZKP being enabled")
        
        # Calculate class weights for imbalanced data
        unique_classes, class_counts = np.unique(self.y_train, return_counts=True)
        if len(unique_classes) > 1:
            # Inverse frequency weighting
            total = np.sum(class_counts)
            class_weights = {
                cls: total / (len(unique_classes) * count)
                for cls, count in zip(unique_classes, class_counts)
            }
            
            # Add weights for all possible classes
            for cls in range(self.num_classes):
                if cls not in class_weights:
                    class_weights[cls] = 1.0
        else:
            class_weights = None
            logger.warning(f"Subject {self.subject_id} has only {unique_classes} class(es) in training data!")
            logger.warning("Model performance may be degraded due to class imbalance")
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                min_delta=0.01
            )
        ]
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=0.2,
            stratify=self.y_train if len(unique_classes) > 1 else None,
            random_state=42
        )
        
        # Configure training
        epochs = config.get("epochs", 10)
        batch_size = config.get("batch_size", 32)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Update history
        self.history["train_loss"].extend(history.history["loss"])
        self.history["train_accuracy"].extend(history.history["accuracy"])
        self.history["val_loss"].extend(history.history["val_loss"])
        self.history["val_accuracy"].extend(history.history["val_accuracy"])
        
        # Apply differential privacy if enabled
        if self.use_differential_privacy:
            weights = self.model.get_weights()
            for i in range(len(weights)):
                noise = np.random.normal(0, self.dp_noise_scale, weights[i].shape)
                weights[i] = weights[i] + noise
            
            self.model.set_weights(weights)
            self.remaining_privacy_budget -= self.dp_epsilon
        
        # Calculate validation metrics
        y_pred = np.argmax(self.model.predict(X_val), axis=1)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        # Prepare metrics (flat dictionary with primitive types only)
        metrics = {
            "train_loss": float(history.history["loss"][-1]),
            "train_accuracy": float(history.history["accuracy"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1]),
            "f1_score": float(f1),
            "dp_budget_remaining": float(self.remaining_privacy_budget),
            "client_id": self.client_id,
            "num_examples": len(self.X_train)
        }
        
        # Handle ZKP proof - Flower only allows primitive types in metrics
        # We need to serialize the proof dictionary appropriately
        if proof and self.use_zero_knowledge_proofs:
            try:
                # Convert dictionary to a flattened version with primitive types
                for key, value in proof.items():
                    # Ensure the key has a prefix to avoid collisions
                    safe_key = f"zkp_{key}"
                    
                    # Handle different types of values that might be in the proof
                    if isinstance(value, (int, float, str, bool)):
                        metrics[safe_key] = value
                    elif isinstance(value, bytes):
                        # Convert bytes to base64 string
                        metrics[safe_key] = base64.b64encode(value).decode('utf-8')
                    elif isinstance(value, list) and all(isinstance(x, (int, float, str, bool, bytes)) for x in value):
                        # Handle lists of primitives
                        if all(isinstance(x, bytes) for x in value):
                            # Convert list of bytes to list of base64 strings
                            metrics[safe_key] = [base64.b64encode(x).decode('utf-8') for x in value]
                        else:
                            metrics[safe_key] = value
                    else:
                        # Convert complex objects to string representation
                        metrics[safe_key] = str(value)
                
                logger.info("Added ZKP proof to metrics (flattened format)")
            except Exception as e:
                logger.error(f"Error serializing ZKP proof: {str(e)}")
                # Add a flag to indicate there was a serialization issue
                metrics["zkp_serialization_error"] = str(e)
        
        # Return updated model weights and metrics
        return self.model.get_weights(), len(X_train), metrics
    
    def evaluate(self, parameters, config):
        """Evaluate the model on local test data."""
        self.model.set_weights(parameters)
        
        # Evaluate model with return_dict for cleaner access to metrics
        eval_results = self.model.evaluate(self.X_test, self.y_test, verbose=0, return_dict=True)
        
        # Get the loss and accuracy values
        loss = eval_results['loss']
        # Choose categorical_accuracy if available, otherwise fallback to accuracy
        accuracy = eval_results.get('categorical_accuracy', eval_results.get('accuracy', 0.0))
        
        # Additional metrics
        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
        try:
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            report = classification_report(self.y_test, y_pred, 
                  output_dict=True, zero_division=0)
            
            # Save confusion matrix for visualization
            self.plot_confusion_matrix(self.y_test, y_pred)
            
            # Save metrics
            metrics = {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "precision": float(report["weighted avg"]["precision"]),
                "recall": float(report["weighted avg"]["recall"]),
                "client_id": self.client_id,
                "num_examples": len(self.X_test)
            }
            
            self.save_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics = {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "error": str(e),
                "client_id": self.client_id,
                "num_examples": len(self.X_test)
            }
        
        return loss, len(self.X_test), metrics
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix."""
        try:
            # Get unique classes
            classes = np.unique(np.concatenate([y_true, y_pred]))
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            
            # Create normalized version
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            
            # Plot
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix (Counts) - S{self.subject_id}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            
            plt.subplot(1, 2, 2)
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
            plt.title(f"Confusion Matrix (Normalized) - S{self.subject_id}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"S{self.subject_id}_confusion_matrix.png"))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
    
    def save_metrics(self, metrics):
        """Save evaluation metrics to CSV."""
        try:
            # Convert metrics to DataFrame
            metrics_df = pd.DataFrame([metrics])
            metrics_df["subject"] = f"S{self.subject_id}"
            metrics_df["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save to CSV
            csv_path = os.path.join(RESULTS_DIR, f"S{self.subject_id}_metrics.csv")
            metrics_df.to_csv(csv_path, index=False)
            logger.info(f"Saved metrics to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")


def main():
    """Main entry point for client."""
    try:
        # Get parameters from command line or environment variables
        params = get_client_params()
        
        # Extract parameters
        subject_id = params.subject_id
        window_size = params.window_size
        subset_size = params.subset_size
        server_address = params.server_address
        use_homomorphic_encryption = params.use_homomorphic_encryption
        use_differential_privacy = params.use_differential_privacy
        use_zero_knowledge_proofs = params.use_zero_knowledge_proofs
        dp_epsilon = params.dp_epsilon
        noise_multiplier = params.noise_multiplier
        privacy_budget = params.privacy_budget
        
        # Log configuration
        logger.info(f"Starting secure client for subject {subject_id} with window size {window_size} and {subset_size*100:.1f}% of data")
        logger.info(f"Security settings - HE: {use_homomorphic_encryption}, DP: {use_differential_privacy}, ZKP: {use_zero_knowledge_proofs}")
        
        # Initialize client
        client = SecureWESADClient(
            subject_id=subject_id,
            window_size=window_size,
            subset_size=subset_size,
            server_address=server_address,
            use_homomorphic_encryption=use_homomorphic_encryption,
            use_differential_privacy=use_differential_privacy,
            use_zero_knowledge_proofs=use_zero_knowledge_proofs,
            dp_epsilon=dp_epsilon,
            noise_multiplier=noise_multiplier,
            privacy_budget=privacy_budget
        )
        
        # Start client with improved timeout settings
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client,
            grpc_max_message_length=24 * 1024 * 1024  # 24MB message size
            
        )
        
        logger.info(f"Client for subject {subject_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Client for subject {subject_id if 'subject_id' in locals() else 'unknown'} failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


# Entry point
if __name__ == "__main__":
    main()