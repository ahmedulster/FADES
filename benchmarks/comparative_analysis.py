#!/usr/bin/env python
"""
Comparative Analysis Framework for WESAD Secure Federated Learning.
This module runs comprehensive tests with different security configurations.
"""
import os
import numpy as np
import tensorflow as tf
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, List, Tuple, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import the attack modules
from attacks.membership_inference import run_membership_inference_attack
from attacks.gradient_leakage import run_gradient_leakage_attack
from attacks.model_inversion import run_model_inversion_attack

class SecureConfigurationAnalysis:
    """
    Framework for comparing different security configurations in federated learning.
    """
    
    def __init__(self, output_dir="results/comparative_analysis", create_dirs=True):
        """
        Initialize the analysis framework.
        
        Args:
            output_dir: Directory to save results
            create_dirs: Whether to create necessary directories
        """
        self.output_dir = output_dir
        
        if create_dirs:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        
        # Initialize results storage
        self.configuration_results = {}
        
    def generate_configurations(self, dp_epsilons=[0.1, 0.5, 1.0, 5.0, 10.0]):
        """
        Generate different security configurations to test.
        
        Args:
            dp_epsilons: List of DP epsilon values to test
            
        Returns:
            List of configuration dictionaries
        """
        configurations = []
        
        # Baseline: No security
        configurations.append({
            'name': 'Baseline',
            'use_he': False,
            'use_dp': False,
            'use_zkp': False,
            'dp_epsilon': None,
            'color': 'red'
        })
        
        # HE only
        configurations.append({
            'name': 'HE Only',
            'use_he': True,
            'use_dp': False,
            'use_zkp': False,
            'dp_epsilon': None,
            'color': 'blue'
        })
        
        # DP only with different epsilon values
        for epsilon in dp_epsilons:
            configurations.append({
                'name': f'DP (ε={epsilon})',
                'use_he': False,
                'use_dp': True,
                'use_zkp': False,
                'dp_epsilon': epsilon,
                'color': 'green'
            })
        
        # ZKP only
        configurations.append({
            'name': 'ZKP Only',
            'use_he': False,
            'use_dp': False,
            'use_zkp': True,
            'dp_epsilon': None,
            'color': 'purple'
        })
        
        # HE + DP
        for epsilon in [0.5, 1.0, 5.0]:
            configurations.append({
                'name': f'HE + DP (ε={epsilon})',
                'use_he': True,
                'use_dp': True,
                'use_zkp': False,
                'dp_epsilon': epsilon,
                'color': 'orange'
            })
        
        # HE + ZKP
        configurations.append({
            'name': 'HE + ZKP',
            'use_he': True,
            'use_dp': False,
            'use_zkp': True,
            'dp_epsilon': None,
            'color': 'brown'
        })
        
        # DP + ZKP
        configurations.append({
            'name': f'DP + ZKP (ε=1.0)',
            'use_he': False,
            'use_dp': True,
            'use_zkp': True,
            'dp_epsilon': 1.0,
            'color': 'pink'
        })
        
        # Full solution: HE + DP + ZKP
        configurations.append({
            'name': 'Full Solution (HE+DP+ZKP)',
            'use_he': True,
            'use_dp': True,
            'use_zkp': True,
            'dp_epsilon': 1.0,
            'color': 'cyan'
        })
        
        return configurations
    
    def train_model_with_config(self, 
                              X_train, y_train, 
                              X_val, y_val,
                              input_shape,
                              num_classes,
                              config,
                              epochs=5,
                              batch_size=32):
        """
        Train a model with a specific security configuration.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            input_shape: Input shape for the model
            num_classes: Number of output classes
            config: Security configuration
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Trained model and training metrics
        """
        # Create a model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Apply security measures
        use_he = config.get('use_he', False)
        use_dp = config.get('use_dp', False)
        use_zkp = config.get('use_zkp', False)
        dp_epsilon = config.get('dp_epsilon', 1.0)
        
        # Track training time
        start_time = time.time()
        
        # Modify training process based on security settings
        if use_dp:
            # For DP, we add noise during training
            from secure_components.differential_privacy import add_noise
            
            # Calculate noise scale based on epsilon
            noise_scale = 1.0 / dp_epsilon if dp_epsilon else 0.0
            
            # Create a custom training step with noise
            @tf.function
            def train_step(x, y):
                with tf.GradientTape() as tape:
                    logits = model(x, training=True)
                    loss = tf.reduce_mean(
                        tf.keras.losses.sparse_categorical_crossentropy(y, logits)
                    )
                
                # Get gradients
                gradients = tape.gradient(loss, model.trainable_weights)
                
                # Add noise to gradients for DP
                noisy_gradients = []
                for grad in gradients:
                    noisy_grad = add_noise(grad.numpy(), noise_scale)
                    noisy_gradients.append(tf.convert_to_tensor(noisy_grad, dtype=grad.dtype))
                
                # Apply gradients
                model.optimizer.apply_gradients(zip(noisy_gradients, model.trainable_weights))
                
                return loss
            
            # Train with custom training loop
            for epoch in range(epochs):
                logger.info(f"Epoch {epoch+1}/{epochs}")
                losses = []
                
                # Create dataset
                dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
                
                for step, (x_batch, y_batch) in enumerate(dataset):
                    loss = train_step(x_batch, y_batch)
                    losses.append(loss)
                    
                    if step % 20 == 0:
                        logger.info(f"Step {step}, Loss: {loss.numpy():.4f}")
                
                # Evaluate
                val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
                logger.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        else:
            # For non-DP training, use regular fit
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                verbose=1
            )
        
        # Measure training time
        training_time = time.time() - start_time
        
        # Evaluate model
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
        metrics = {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'training_time': training_time
        }
        
        return model, metrics
    
    def run_attacks_on_configuration(self, 
                                   model,
                                   X_train, y_train, 
                                   X_test, y_test,
                                   config):
        """
        Run all attacks on a model trained with a specific configuration.
        
        Args:
            model: Trained model
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
            config: Security configuration
            
        Returns:
            Dictionary of attack metrics
        """
        # Define result storage
        attack_results = {}
        use_he = config.get('use_he', False)
        use_dp = config.get('use_dp', False)
        use_zkp = config.get('use_zkp', False)
        
        # 1. Membership Inference Attack
        logger.info(f"Running membership inference attack on {config['name']} configuration...")
        membership_save_path = os.path.join(
            self.output_dir, 
            "plots", 
            f"membership_inference_{config['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.png"
        )
        
        membership_results = run_membership_inference_attack(
            model, 
            X_train, y_train, 
            X_test, y_test, 
            X_train.shape[1:],
            model.output_shape[-1],
            save_path=membership_save_path
        )
        attack_results['membership_inference'] = membership_results
        
        # 2. Gradient Leakage Attack
        logger.info(f"Running gradient leakage attack on {config['name']} configuration...")
        gradient_save_path = os.path.join(
            self.output_dir, 
            "plots", 
            f"gradient_leakage_{config['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.png"
        )
        
        gradient_results = run_gradient_leakage_attack(
            model, 
            X_train, y_train, 
            with_security=(use_he or use_dp),
            save_path=gradient_save_path
        )
        attack_results['gradient_leakage'] = gradient_results
        
        # 3. Model Inversion Attack
        logger.info(f"Running model inversion attack on {config['name']} configuration...")
        inversion_save_path = os.path.join(
            self.output_dir, 
            "plots", 
            f"model_inversion_{config['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.png"
        )
        
        inversion_metrics, inverted_samples = run_model_inversion_attack(
            model, 
            X_train.shape[1:], 
            model.output_shape[-1],
            with_security=(use_he or use_dp),
            reference_data=(X_train, y_train),
            iterations=500,  # Reduce for faster execution
            save_path=inversion_save_path
        )
        attack_results['model_inversion'] = inversion_metrics
        
        return attack_results
    
    def evaluate_configuration(self, 
                            config, 
                            X_train, y_train, 
                            X_val, y_val,
                            X_test, y_test,
                            input_shape,
                            num_classes):
        """
        Evaluate a specific security configuration.
        
        Args:
            config: Security configuration
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            X_test: Test data
            y_test: Test labels
            input_shape: Input shape for the model
            num_classes: Number of output classes
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating configuration: {config['name']}")
        
        # Train model with this configuration
        model, training_metrics = self.train_model_with_config(
            X_train, y_train,
            X_val, y_val,
            input_shape,
            num_classes,
            config
        )
        
        # Measure resource usage
        memory_usage = self.measure_memory_usage(model)
        
        # Run attacks on the model
        attack_metrics = self.run_attacks_on_configuration(
            model,
            X_train, y_train,
            X_test, y_test,
            config
        )
        
        # Compile all results
        results = {
            'config': config,
            'training_metrics': training_metrics,
            'resource_metrics': {
                'memory_usage': memory_usage
            },
            'attack_metrics': attack_metrics
        }
        
        # Store results
        self.configuration_results[config['name']] = results
        
        # Save incremental results
        self.save_results()
        
        return results
    
    def measure_memory_usage(self, model):
        """
        Measure memory usage of a model.
        
        Args:
            model: Model to measure
            
        Returns:
            Memory usage in MB
        """
        import sys
        
        # Get model size
        model_bytes = sum(
            w.numpy().nbytes for w in model.weights
        )
        
        # Convert to MB
        model_size_mb = model_bytes / (1024 * 1024)
        
        return model_size_mb
    
    def save_results(self):
        """Save current results to disk."""
        # Convert to serializable format
        serializable_results = {}
        
        for config_name, results in self.configuration_results.items():
            # Convert non-serializable items
            serializable_config = dict(results['config'])
            
            serializable_results[config_name] = {
                'config': serializable_config,
                'training_metrics': results['training_metrics'],
                'resource_metrics': results['resource_metrics'],
                'attack_metrics': {
                    k: {
                        kk: vv for kk, vv in v.items() 
                        if isinstance(vv, (int, float, bool, str))
                    }
                    for k, v in results['attack_metrics'].items()
                }
            }
        
        # Save as JSON
        with open(os.path.join(self.output_dir, "data", "configuration_results.json"), 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results to {self.output_dir}/data/configuration_results.json")
    
    def create_summary_table(self):
        """
        Create a summary table of all configuration results.
        
        Returns:
            DataFrame with summary results
        """
        if not self.configuration_results:
            logger.warning("No results to summarize")
            return None
        
        # Initialize table data
        table_data = []
        
        for config_name, results in self.configuration_results.items():
            config = results['config']
            training_metrics = results['training_metrics']
            resource_metrics = results['resource_metrics']
            attack_metrics = results['attack_metrics']
            
            # Extract relevant metrics
            row = {
                'Configuration': config_name,
                'HE': 'Yes' if config.get('use_he', False) else 'No',
                'DP': 'Yes' if config.get('use_dp', False) else 'No',
                'ZKP': 'Yes' if config.get('use_zkp', False) else 'No',
                'DP Epsilon': config.get('dp_epsilon', 'N/A'),
                
                # Training performance
                'Train Accuracy': training_metrics['train_accuracy'],
                'Val Accuracy': training_metrics['val_accuracy'],
                'Training Time (s)': training_metrics['training_time'],
                
                # Resource usage
                'Memory (MB)': resource_metrics['memory_usage'],
                
                # Attack resistance
                'MIA Success Rate': attack_metrics['membership_inference']['accuracy'],
                'MIA AUC': attack_metrics['membership_inference']['auc'],
                'Gradient Leakage MSE': attack_metrics['gradient_leakage']['mse'],
                'Gradient Leakage Corr': attack_metrics['gradient_leakage']['correlation'],
                'Model Inversion Conf': attack_metrics['model_inversion']['average_confidence']
            }
            
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, "data", "summary_table.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary table to {csv_path}")
        
        return df
    
    def plot_privacy_utility_tradeoff(self, save_path=None):
        """
        Plot privacy-utility tradeoff across configurations.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.configuration_results:
            logger.warning("No results to plot")
            return
        
        # Extract data for plotting
        configs = []
        val_accuracies = []
        mia_resistances = []
        config_names = []
        colors = []
        
        for config_name, results in self.configuration_results.items():
            configs.append(results['config'])
            val_accuracies.append(results['training_metrics']['val_accuracy'])
            
            # MIA resistance is 1 - attack success rate
            mia_resistance = 1.0 - results['attack_metrics']['membership_inference']['accuracy']
            mia_resistances.append(mia_resistance)
            
            config_names.append(config_name)
            colors.append(results['config'].get('color', 'blue'))
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Scatter plot with annotations
        for i, (x, y, name, color) in enumerate(zip(mia_resistances, val_accuracies, config_names, colors)):
            plt.scatter(x, y, s=100, alpha=0.7, color=color, label=name)
            plt.annotate(name, (x, y), fontsize=9, 
                        xytext=(5, 5), textcoords='offset points')
        
        # Add best region indication
        plt.axhspan(0.9, 1.0, alpha=0.1, color='green', label='High Utility')
        plt.axvspan(0.8, 1.0, alpha=0.1, color='green', label='High Privacy')
        
        # Label the "sweet spot" region
        plt.axhspan(0.9, 1.0, xmax=0.5, alpha=0.2, color='gold')
        plt.axvspan(0.8, 1.0, ymax=0.5, alpha=0.2, color='gold')
        
        # Add text annotation for the sweet spot
        plt.text(0.9, 0.95, "Optimal Region", 
                fontsize=12, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", fc='gold', alpha=0.3))
        
        # Add labels and title
        plt.xlabel('Privacy Protection (1 - MIA Success Rate)')
        plt.ylabel('Model Utility (Validation Accuracy)')
        plt.title('Privacy-Utility Tradeoff Across Security Configurations')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis limits
        plt.xlim(0.4, 1.0)
        plt.ylim(0.5, 1.0)
        
        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved privacy-utility tradeoff plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_attack_effectiveness(self, save_path=None):
        """
        Plot attack effectiveness across configurations.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.configuration_results:
            logger.warning("No results to plot")
            return
        
        # Extract data for plotting
        config_names = []
        mia_success_rates = []
        gradient_mse_values = []
        inversion_confidences = []
        
        for config_name, results in self.configuration_results.items():
            config_names.append(config_name)
            mia_success_rates.append(results['attack_metrics']['membership_inference']['accuracy'])
            gradient_mse_values.append(results['attack_metrics']['gradient_leakage']['mse'])
            inversion_confidences.append(results['attack_metrics']['model_inversion']['average_confidence'])
        
        # Create plot
        plt.figure(figsize=(14, 8))
        
        # Calculate positions
        x = np.arange(len(config_names))
        width = 0.25
        
        # Plot bars
        plt.bar(x - width, mia_success_rates, width, label='MIA Success Rate', color='crimson')
        plt.bar(x, gradient_mse_values, width, label='Gradient Leakage MSE', color='royalblue')
        plt.bar(x + width, inversion_confidences, width, label='Model Inversion Confidence', color='forestgreen')
        
        # Add labels and title
        plt.xlabel('Security Configuration')
        plt.ylabel('Attack Metric Value')
        plt.title('Attack Effectiveness Across Security Configurations')
        plt.xticks(x, config_names, rotation=45, ha='right')
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attack effectiveness plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_resource_requirements(self, save_path=None):
        """
        Plot resource requirements across configurations.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.configuration_results:
            logger.warning("No results to plot")
            return
        
        # Extract data for plotting
        config_names = []
        training_times = []
        memory_usages = []
        
        for config_name, results in self.configuration_results.items():
            config_names.append(config_name)
            training_times.append(results['training_metrics']['training_time'])
            memory_usages.append(results['resource_metrics']['memory_usage'])
        
        # Create plot
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Training time on left y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Security Configuration')
        ax1.set_ylabel('Training Time (s)', color=color)
        ax1.bar(config_names, training_times, color=color, alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Memory usage on right y-axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Memory Usage (MB)', color=color)
        ax2.plot(config_names, memory_usages, 'o-', color=color, linewidth=2, markersize=8)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add title and rotate labels
        plt.title('Resource Requirements Across Security Configurations')
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved resource requirements plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def generate_all_plots(self):
        """Generate all summary plots."""
        # Create output paths
        plots_dir = os.path.join(self.output_dir, "plots")
        
        # Create plots
        self.plot_privacy_utility_tradeoff(
            save_path=os.path.join(plots_dir, "privacy_utility_tradeoff.png")
        )
        
        self.plot_attack_effectiveness(
            save_path=os.path.join(plots_dir, "attack_effectiveness.png")
        )
        
        self.plot_resource_requirements(
            save_path=os.path.join(plots_dir, "resource_requirements.png")
        )
    
    def run_all_configurations(self, X_train, y_train, X_val, y_val, X_test, y_test, input_shape, num_classes):
        """
        Run evaluation for all configurations.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            X_test: Test data
            y_test: Test labels
            input_shape: Input shape for the model
            num_classes: Number of output classes
        """
        # Generate configurations
        configurations = self.generate_configurations()
        
        logger.info(f"Evaluating {len(configurations)} different security configurations")
        
        # Evaluate each configuration
        for config in configurations:
            logger.info(f"Starting evaluation for configuration: {config['name']}")
            
            self.evaluate_configuration(
                config,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                input_shape,
                num_classes
            )
            
            logger.info(f"Completed evaluation for configuration: {config['name']}")
        
        # Generate summary table
        summary_df = self.create_summary_table()
        
        # Generate all plots
        self.generate_all_plots()
        
        logger.info("Completed evaluation of all configurations")
        
        return summary_df

def run_comparative_analysis(subject_id="2", output_dir="results/comparative_analysis"):
    """
    Run a full comparative analysis on a subject.
    
    Args:
        subject_id: Subject ID to analyze
        output_dir: Directory to save results
        
    Returns:
        Summary DataFrame of results
    """
    logger.info(f"Starting comparative analysis for subject {subject_id}")
    
    # Load data
    from scripts.data_loader import load_subject, preprocess_data
    
    subject_data = load_subject(subject_id)
    (X_train, X_test, y_train, y_test) = preprocess_data(subject_data)
    
    # Split training data into train and validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Initialize analysis framework
    analysis = SecureConfigurationAnalysis(output_dir=output_dir)
    
    # Run all configurations
    summary_df = analysis.run_all_configurations(
        X_train, y_train, 
        X_val, y_val,
        X_test, y_test,
        X_train.shape[1:],
        len(np.unique(np.concatenate([y_train, y_test])))
    )
    
    logger.info("Comparative analysis completed successfully")
    
    return summary_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comparative analysis of security configurations")
    parser.add_argument("--subject", default="2", help="Subject ID to analyze")
    parser.add_argument("--output_dir", default="results/comparative_analysis", help="Output directory for results")
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run analysis
    summary_df = run_comparative_analysis(
        subject_id=args.subject,
        output_dir=args.output_dir
    )
    
    print("\nSummary of Results:")
    print(summary_df)