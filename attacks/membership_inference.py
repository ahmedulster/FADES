#!/usr/bin/env python
"""
Membership Inference Attack implementation for WESAD Federated Learning.
Tests whether an adversary can determine if a specific data point was used during training.
"""
import os
import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("results", exist_ok=True)

class MembershipInferenceAttack:
    """
    Implements a simplified membership inference attack using confidence values.
    Instead of creating shadow models, this implementation uses prediction confidence
    thresholds, which is simpler to implement but still effective.
    """
    
    def __init__(self, model):
        """
        Initialize the attack with a model to attack.
        
        Args:
            model: The model to attack
        """
        self.model = model
    
    def get_confidence_values(self, X, y):
        """
        Get model confidence values for data points.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Confidence values for true class
        """
        # Get model predictions
        predictions = self.model.predict(X)
        
        # If y is one-hot encoded, convert to indices
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_indices = np.argmax(y, axis=1)
        else:
            y_indices = y
        
        # Get confidence for the true class for each sample
        confidences = np.array([predictions[i, y_indices[i]] for i in range(len(y_indices))])
        
        return confidences
    
    def evaluate_attack(self, members_X, members_y, nonmembers_X, nonmembers_y):
        """
        Evaluate the attack using member and non-member data.
        
        Args:
            members_X: Data that was used during training
            members_y: Labels for training data
            nonmembers_X: Data that was not used during training
            nonmembers_y: Labels for non-training data
            
        Returns:
            Dictionary with attack metrics
        """
        # Get confidence values
        member_confidences = self.get_confidence_values(members_X, members_y)
        nonmember_confidences = self.get_confidence_values(nonmembers_X, nonmembers_y)
        
        # Combine confidences and create labels (1 for members, 0 for non-members)
        all_confidences = np.concatenate([member_confidences, nonmember_confidences])
        all_labels = np.concatenate([np.ones_like(member_confidences), np.zeros_like(nonmember_confidences)])
        
        # Find best threshold using ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_confidences)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold for classification
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Make predictions using the threshold
        predictions = (all_confidences >= optimal_threshold).astype(int)
        attack_accuracy = accuracy_score(all_labels, predictions)
        
        # Calculate precision and recall
        true_positives = np.sum((predictions == 1) & (all_labels == 1))
        false_positives = np.sum((predictions == 1) & (all_labels == 0))
        true_negatives = np.sum((predictions == 0) & (all_labels == 0))
        false_negatives = np.sum((predictions == 0) & (all_labels == 1))
        
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-6)
        
        # Compare confidence distributions
        logger.info(f"Member confidence: mean={np.mean(member_confidences):.4f}, std={np.std(member_confidences):.4f}")
        logger.info(f"Non-member confidence: mean={np.mean(nonmember_confidences):.4f}, std={np.std(nonmember_confidences):.4f}")
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"Attack accuracy: {attack_accuracy:.4f}, AUC: {roc_auc:.4f}")
        
        # Return metrics
        return {
            'accuracy': float(attack_accuracy),
            'auc': float(roc_auc),
            'threshold': float(optimal_threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'member_confidences': member_confidences,
            'nonmember_confidences': nonmember_confidences
        }
    
    def plot_attack_results(self, metrics, save_path=None):
        """
        Plot attack results.
        
        Args:
            metrics: Dictionary of attack metrics
            save_path: Path to save the plot
        """
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Confidence distributions
        plt.subplot(1, 2, 1)
        plt.hist(metrics['member_confidences'], bins=20, alpha=0.5, label='Members')
        plt.hist(metrics['nonmember_confidences'], bins=20, alpha=0.5, label='Non-members')
        plt.axvline(metrics['threshold'], color='r', linestyle='--', label=f'Threshold ({metrics["threshold"]:.3f})')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Key metrics
        plt.subplot(1, 2, 2)
        metrics_to_plot = ['accuracy', 'auc', 'precision', 'recall', 'f1_score']
        values = [metrics[m] for m in metrics_to_plot]
        
        bar_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
        plt.bar(metrics_to_plot, values, color=bar_colors)
        plt.ylim(0, 1.1)  # All metrics are between 0 and 1
        plt.ylabel('Score')
        plt.title('Attack Performance Metrics')
        
        # Add value labels on top of bars
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
            
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attack results to {save_path}")
            
        plt.close()

def run_membership_inference_attack(
    model, 
    X_train, y_train, 
    X_test, y_test, 
    input_shape=None,  # Not needed for this implementation but kept for API compatibility
    num_classes=None,  # Not needed for this implementation but kept for API compatibility
    save_path=None
):
    """
    Run a membership inference attack evaluation.
    
    Args:
        model: Model to attack
        X_train: Training data features
        y_train: Training data labels
        X_test: Test data features
        y_test: Test data labels
        input_shape: Shape of input data (not used in this implementation)
        num_classes: Number of output classes (not used in this implementation)
        save_path: Path to save results plot
        
    Returns:
        Dictionary of attack results
    """
    try:
        logger.info("Initializing membership inference attack...")
        
        # Initialize attack
        attack = MembershipInferenceAttack(model)
        
        # Measure attack time
        start_time = time.time()
        
        # For simplicity and clearer evaluation, we'll use:
        # - Half of training data as "members"
        # - Half of test data as "non-members"
        # This ensures we're testing against data the model has actually seen
        
        train_indices = np.random.choice(len(X_train), min(len(X_train) // 2, 100), replace=False)
        test_indices = np.random.choice(len(X_test), min(len(X_test) // 2, 100), replace=False)
        
        members_X = X_train[train_indices]
        members_y = y_train[train_indices]
        nonmembers_X = X_test[test_indices]
        nonmembers_y = y_test[test_indices]
        
        logger.info(f"Running attack with {len(members_X)} members and {len(nonmembers_X)} non-members")
        
        # Evaluate attack
        metrics = attack.evaluate_attack(members_X, members_y, nonmembers_X, nonmembers_y)
        
        # Add attack time
        elapsed_time = time.time() - start_time
        metrics['attack_time'] = elapsed_time
        logger.info(f"Attack completed in {elapsed_time:.2f} seconds")
        
        # Plot results
        if save_path:
            attack.plot_attack_results(metrics, save_path)
        
        # Return results (excluding confidence arrays to keep the result size manageable)
        result_metrics = {k: v for k, v in metrics.items() 
                         if k not in ['member_confidences', 'nonmember_confidences']}
        
        logger.info(f"Membership Inference Attack Results: Accuracy={result_metrics['accuracy']:.4f}, AUC={result_metrics['auc']:.4f}")
        return result_metrics
        
    except Exception as e:
        logger.error(f"Error in membership inference attack: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return error metrics
        return {
            'error': str(e),
            'accuracy': 0.5,  # Random guessing
            'auc': 0.5,       # Random guessing
            'attack_time': 0.0
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
    
    logger.info("Loading data for testing membership inference attack...")
    
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
        model.fit(X_train[:100], y_train[:100], epochs=3, batch_size=32, verbose=1)
        
        # Run the attack
        results = run_membership_inference_attack(
            model, X_train[:100], y_train[:100], X_test, y_test, 
            save_path="results/membership_inference_attack_results.png"
        )
        
        # Print results
        logger.info("\nAttack Results:")
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
                
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())