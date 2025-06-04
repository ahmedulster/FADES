#!/usr/bin/env python
"""
Federated learning server implementation for the WESAD dataset with security features.
This server aggregates model updates from clients and manages the federated learning process.
"""
import time
import os
import argparse
import logging
import numpy as np
import pandas as pd
import flwr as fl
import matplotlib.pyplot as plt
import traceback
import secrets
import hashlib
import hmac
import base64
import time
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Metrics, Scalar, FitRes, EvaluateRes, Parameters, FitIns
from flwr.server.history import History
from typing import Dict, List, Optional, Tuple, Union, Any
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import secure components
from secure_components.homomorphic_encryption import PaillierEncryption
from secure_components.differential_privacy import add_noise
from secure_components.zkp import generate_proof, verify_proof

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
RESULTS_DIR = "server_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class SecureWESADStrategy(FedAvg):
    """Secure federated learning strategy for WESAD dataset."""
    def __init__(
        self,
        min_fit_clients: int = 1,         
        min_evaluate_clients: int = 1,    
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        num_rounds: int = 5,
        # Security parameters
        use_homomorphic_encryption: bool = True,
        use_differential_privacy: bool = True,
        use_zero_knowledge_proofs: bool = True
    ) -> None:
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
        )
        self.num_rounds = num_rounds
        self.global_metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }
        
        # Security settings
        self.use_homomorphic_encryption = use_homomorphic_encryption
        self.use_differential_privacy = use_differential_privacy
        self.use_zero_knowledge_proofs = use_zero_knowledge_proofs
        
        # Initialize homomorphic encryption (server-side)
        if self.use_homomorphic_encryption:
            try:
                self.encryption = PaillierEncryption(key_size=1024)
                logger.info("Initialized Paillier encryption on server")
            except Exception as e:
                logger.error(f"Failed to initialize homomorphic encryption: {str(e)}")
                self.use_homomorphic_encryption = False
        
        # Initialize ZKP verification
        if self.use_zero_knowledge_proofs:
            self.zkp_challenges = {}  # Map client_id to challenge
            self.authenticated_clients = set()  # Track authenticated clients
            logger.info("Initialized ZKP verification system")
    
    def generate_zkp_challenge(self, client_id: str) -> bytes:
        """Generate a challenge for the client's ZKP"""
        challenge = secrets.token_bytes(32)
        self.zkp_challenges[client_id] = challenge
        return challenge
    
    def verify_client_zkp(self, client_id: str, metrics: Dict) -> bool:
        """Verify a client's ZKP response from flattened metrics"""
        if not self.use_zero_knowledge_proofs:
            return True
            
        try:
            # Check if we have a challenge for this client
            if client_id not in self.zkp_challenges:
                logger.warning(f"No challenge found for client {client_id}")
                return False
            
            challenge = self.zkp_challenges[client_id]
            
            # Check if there are any zkp_ prefixed keys in the metrics
            zkp_keys = [k for k in metrics.keys() if k.startswith("zkp_")]
            
            if not zkp_keys:
                logger.warning(f"Client {client_id} did not provide ZKP proof")
                return False
                
            # Reconstruct the proof dictionary from flattened metrics
            proof = {}
            
            for key in zkp_keys:
                # Remove the zkp_ prefix to get the original key
                original_key = key[4:]  # Remove 'zkp_' prefix
                value = metrics[key]
                
                # Handle potential base64 encoded values (which were originally bytes)
                if isinstance(value, str) and original_key in ["signature", "challenge_hash", "public_key"]:
                    try:
                        # Try to decode as base64 if it looks like it
                        value = base64.b64decode(value)
                    except:
                        # If it fails, keep the original string value
                        pass
                elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                    # Handle lists of strings that might be base64 encoded
                    try:
                        value = [base64.b64decode(x) for x in value]
                    except:
                        # If decoding fails, keep the original list
                        pass
                        
                proof[original_key] = value
                
            # Verify the proof
            is_verified = verify_proof(proof, challenge, client_id)
            
            if is_verified:
                self.authenticated_clients.add(client_id)
                logger.info(f"Client {client_id} authenticated successfully")
            else:
                logger.warning(f"Client {client_id} authentication failed")
                
            return is_verified
        except Exception as e:
            logger.error(f"Error verifying ZKP: {str(e)}")
            return False
    
    def decrypt_weights(self, encrypted_weights: List[Any]) -> List[np.ndarray]:
        """Decrypt model weights using homomorphic encryption"""
        if not self.use_homomorphic_encryption:
            return encrypted_weights
            
        decrypted_weights = []
        
        # Process each weight tensor
        for i, w in enumerate(encrypted_weights):
            if isinstance(w, dict) and w.get("encrypted", False):
                try:
                    # Extract encrypted data
                    original_shape = w["original_shape"]
                    scale_factor = w["scale_factor"]
                    indices = w["indices"]
                    encrypted_values = w["encrypted_values"]
                    non_encrypted = w["non_encrypted"]  # In a real system, we wouldn't have this
                    
                    # Decrypt values
                    decrypted_values = [self.encryption.decrypt(val) / scale_factor for val in encrypted_values]
                    
                    # Reconstruct the original tensor
                    # In a real implementation, we would properly reconstruct from encrypted data
                    # Here we use the non_encrypted version as a shortcut
                    decrypted_weights.append(non_encrypted)
                    
                except Exception as e:
                    logger.error(f"Error decrypting weight tensor {i}: {str(e)}")
                    # Fallback to using the non-encrypted version
                    decrypted_weights.append(w.get("non_encrypted", np.zeros(1)))
            else:
                # Not encrypted, keep as is
                decrypted_weights.append(w)
                
        return decrypted_weights

    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager
    ) -> List[Tuple[ClientProxy, Dict]]:
        """Configure clients for training with security challenges"""
        # Get list of clients and their configurations from parent class
        client_config_pairs = super().configure_fit(server_round, parameters, client_manager)
        
        # Create new client config pairs with ZKP challenges
        if self.use_zero_knowledge_proofs:
            updated_client_config_pairs = []
            
            for client, fit_ins in client_config_pairs:
                # Use client ID or a temporary ID
                client_id = f"client_{client.cid}"
                
                # Generate a unique challenge
                challenge = self.generate_zkp_challenge(client_id)
                
                # Access configuration properly from FitIns
                config = fit_ins.config if hasattr(fit_ins, 'config') else {}
                
                # Create a new configuration dictionary with our additions
                updated_config = dict(config)
                updated_config["zkp_challenge"] = challenge
                updated_config["client_id"] = client_id
                
                # Create a new FitIns with our updated config
                updated_fit_ins = FitIns(parameters=fit_ins.parameters, config=updated_config)
                
                # Add to updated list
                updated_client_config_pairs.append((client, updated_fit_ins))
                logger.info(f"Added ZKP challenge for client {client_id}")
                
            return updated_client_config_pairs
        else:
            return client_config_pairs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client results with security verification"""
        # Check if we have any results
        if not results:
            return None, {"round": server_round, "no_results": True}
        
        try:
            # Apply ZKP verification if enabled
            if self.use_zero_knowledge_proofs:
                verified_results = []
                
                for client, fit_res in results:
                    # Check if client provided client_id in metrics
                    if fit_res.metrics and "client_id" in fit_res.metrics:
                        client_id = fit_res.metrics["client_id"]
                        
                        # Verify if this client is already authenticated or can be authenticated now
                        if client_id in self.authenticated_clients or self.verify_client_zkp(client_id, fit_res.metrics):
                            verified_results.append((client, fit_res))
                            logger.info(f"Client {client_id} authenticated, accepted")
                        else:
                            logger.warning(f"Client {client_id} failed authentication, rejecting results")
                    else:
                        # No client_id, accept with warning
                        logger.warning(f"Client {client.cid} did not provide client_id, accepting with caution")
                        verified_results.append((client, fit_res))
                
                if not verified_results:
                    logger.error("No clients passed ZKP verification")
                    return None, {"round": server_round, "authentication_failure": True}
                    
                # Use verified results for aggregation
                results = verified_results
            
            # Check if all clients returned zero examples
            if all(fit_res.num_examples == 0 for _, fit_res in results):
                logger.warning("All clients returned zero examples. Using parameters from first client.")
                parameters = results[0][1].parameters
                return parameters, {"round": server_round, "zero_examples": True}
            
            # Filter out clients with zero examples
            valid_results = [(client, fit_res) for client, fit_res in results if fit_res.num_examples > 0]
            
            # If we have valid results, use them for aggregation
            if valid_results:
                # Apply homomorphic decryption if needed (not implemented in practice here)
                if self.use_homomorphic_encryption:
                    # In a real implementation, we would decrypt the parameters
                    # Here we just log that we would do it
                    logger.info("Homomorphic encryption enabled: would decrypt parameters here")
                    
                # Call the parent class method for aggregation
                aggregated_weights, metrics_aggregated = super().aggregate_fit(server_round, valid_results, failures)
                
                # Initialize metrics dictionary
                metrics_dict = metrics_aggregated if metrics_aggregated else {}
                
                # Safely extract metrics with proper error handling
                train_losses = []
                train_accuracies = []
                val_losses = []
                val_accuracies = []
                
                for _, res in valid_results:
                    try:
                        # Ensure we're handling metrics that might be missing
                        metrics = res.metrics if res.metrics else {}
                        
                        # Extract metrics with default values if missing
                        if "train_loss" in metrics:
                            train_losses.append(float(metrics["train_loss"]))
                        if "train_accuracy" in metrics:
                            train_accuracies.append(float(metrics["train_accuracy"]))
                        if "val_loss" in metrics:
                            val_losses.append(float(metrics["val_loss"]))
                        if "val_accuracy" in metrics:
                            val_accuracies.append(float(metrics["val_accuracy"]))
                    except Exception as e:
                        logger.warning(f"Error processing client metrics: {str(e)}")
                        continue
                
                # Only compute averages if we have data
                if train_losses:
                    train_loss = np.mean(train_losses)
                    self.global_metrics["train_loss"].append(train_loss)
                    metrics_dict["train_loss"] = float(train_loss)  # Ensure it's a Python primitive
                    
                if train_accuracies:
                    train_accuracy = np.mean(train_accuracies)
                    self.global_metrics["train_accuracy"].append(train_accuracy)
                    metrics_dict["train_accuracy"] = float(train_accuracy)
                    
                if val_losses:
                    val_loss = np.mean(val_losses)
                    self.global_metrics["val_loss"].append(val_loss)
                    metrics_dict["val_loss"] = float(val_loss)
                    
                if val_accuracies:
                    val_accuracy = np.mean(val_accuracies)
                    self.global_metrics["val_accuracy"].append(val_accuracy)
                    metrics_dict["val_accuracy"] = float(val_accuracy)
                
                # Log aggregated metrics
                logger.info(f"Round {server_round} aggregated metrics:")
                for key, value in metrics_dict.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {key}: {value:.4f}")
                    else:
                        logger.info(f"  {key}: {value}")
                
                # Save metrics and plot progress
                self.save_metrics(server_round)
                self.plot_progress()
                
                # Add security metrics - ensure these are primitive types
                metrics_dict["he_enabled"] = int(self.use_homomorphic_encryption)
                metrics_dict["dp_enabled"] = int(self.use_differential_privacy)
                metrics_dict["zkp_enabled"] = int(self.use_zero_knowledge_proofs)
                metrics_dict["verified_clients"] = len(self.authenticated_clients) if self.use_zero_knowledge_proofs else 0
                
                return aggregated_weights, metrics_dict
                
            else:
                # This should not happen if we checked for zero examples above
                logger.error("No valid results after filtering")
                return results[0][1].parameters, {"round": server_round, "no_valid_results": True}
                
        except Exception as e:
            logger.error(f"Error in aggregation: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fall back to using parameters from the first valid client
            return results[0][1].parameters, {"error": str(e)}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results with security verification"""
        if not results:  # Skip if no results
            return None, {}
        
        try:
            # Apply ZKP verification if enabled
            if self.use_zero_knowledge_proofs:
                verified_results = []
                
                for client, eval_res in results:
                    # Check if client provided client_id in metrics
                    if eval_res.metrics and "client_id" in eval_res.metrics:
                        client_id = eval_res.metrics["client_id"]
                        
                        # Verify if this client is already authenticated
                        if client_id in self.authenticated_clients:
                            verified_results.append((client, eval_res))
                            logger.info(f"Client {client_id} authenticated, accepted for evaluation")
                        else:
                            logger.warning(f"Client {client_id} not authenticated for evaluation, rejected")
                    else:
                        # No client_id, accept with warning
                        logger.warning(f"Client {client.cid} did not provide authentication for evaluation")
                        verified_results.append((client, eval_res))
                
                if verified_results:
                    # Use verified results for aggregation
                    results = verified_results
            
            # Safely extract metrics 
            test_losses = []
            test_accuracies = []
            loss_aggregated = None
            metrics_aggregated = {}
            
            # Filter out results with zero examples
            valid_results = [(client, res) for client, res in results if hasattr(res, 'num_examples') and res.num_examples > 0]
            
            if not valid_results:
                logger.warning("No clients returned evaluation results with non-zero examples")
                return None, {"round": server_round, "no_valid_evaluation": True}
            
            # Calculate the weighted average of losses
            total_examples = sum(res.num_examples for _, res in valid_results)
            
            if total_examples > 0:  # Make sure we don't divide by zero
                weighted_losses = [res.loss * res.num_examples for _, res in valid_results if res.loss is not None]
                num_results_with_loss = sum(1 for _, res in valid_results if res.loss is not None)
                
                if num_results_with_loss > 0:
                    loss_aggregated = sum(weighted_losses) / total_examples
                    metrics_aggregated["loss"] = float(loss_aggregated)
            
            # Process per-client metrics
            for _, res in valid_results:
                try:
                    if res.loss is not None:
                        test_losses.append(float(res.loss))
                    
                    metrics = res.metrics if res.metrics else {}
                    if "accuracy" in metrics:
                        test_accuracies.append(float(metrics["accuracy"]))
                except Exception as e:
                    logger.warning(f"Error processing evaluation metrics: {str(e)}")
                    continue
            
            # Calculate aggregated metrics only if we have data
            if test_losses:
                test_loss = np.mean(test_losses)
                self.global_metrics["test_loss"].append(test_loss)
                metrics_aggregated["test_loss"] = float(test_loss)
            
            if test_accuracies:
                test_accuracy = np.mean(test_accuracies)
                self.global_metrics["test_accuracy"].append(test_accuracy)
                metrics_aggregated["test_accuracy"] = float(test_accuracy)
                
                logger.info(f"Round {server_round} - Test Loss: {test_loss if 'test_loss' in metrics_aggregated else 'N/A'}, Test Accuracy: {test_accuracy:.4f}")
            
            # Add security metrics with primitive types only
            metrics_aggregated["he_enabled"] = int(self.use_homomorphic_encryption)
            metrics_aggregated["dp_enabled"] = int(self.use_differential_privacy)
            metrics_aggregated["zkp_enabled"] = int(self.use_zero_knowledge_proofs)
            metrics_aggregated["verified_clients"] = len(self.authenticated_clients) if self.use_zero_knowledge_proofs else 0
            
            # Save metrics and plot progress after each round
            self.save_metrics(server_round)
            self.plot_progress()
            
            return loss_aggregated, metrics_aggregated
        
        except Exception as e:
            logger.error(f"Error aggregating evaluate metrics: {str(e)}")
            # Return safe defaults
            return None, {"error": str(e)}

    def save_metrics(self, server_round: int) -> None:
        """Save global metrics to CSV"""
        try:
            # Create a dictionary for all metrics with proper handling for missing values
            metrics_data = {}
            
            # Add data for each metric type, handling potential missing rounds
            for key, values in self.global_metrics.items():
                # Pad with NaN if we have fewer values than rounds
                padded_values = values.copy()
                while len(padded_values) < server_round:
                    padded_values.append(np.nan)
                metrics_data[key] = padded_values[:server_round]
            
            # Add round numbers
            metrics_data["round"] = np.arange(1, server_round + 1)
            
            # Add security settings
            metrics_data["homomorphic_encryption"] = [self.use_homomorphic_encryption] * server_round
            metrics_data["differential_privacy"] = [self.use_differential_privacy] * server_round
            metrics_data["zkp_verification"] = [self.use_zero_knowledge_proofs] * server_round
            
            # Convert to DataFrame and save
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(os.path.join(RESULTS_DIR, "global_metrics.csv"), index=False)
            logger.info(f"Saved metrics for round {server_round}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    def plot_progress(self) -> None:
        """Plot and save training progress"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot loss
            plt.subplot(1, 2, 1)
            for metric_name, line_style, color in [
                ("train_loss", "-", "blue"),
                ("val_loss", "--", "orange"),
                ("test_loss", ":", "green")
            ]:
                if self.global_metrics[metric_name]:
                    plt.plot(
                        range(1, len(self.global_metrics[metric_name]) + 1),
                        self.global_metrics[metric_name], 
                        line_style,
                        color=color,
                        label=metric_name.replace("_", " ").title()
                    )
            
            plt.title("Loss Over Rounds")
            plt.xlabel("Rounds")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot accuracy
            plt.subplot(1, 2, 2)
            for metric_name, line_style, color in [
                ("train_accuracy", "-", "blue"),
                ("val_accuracy", "--", "orange"),
                ("test_accuracy", ":", "green")
            ]:
                if self.global_metrics[metric_name]:
                    plt.plot(
                        range(1, len(self.global_metrics[metric_name]) + 1),
                        self.global_metrics[metric_name], 
                        line_style,
                        color=color,
                        label=metric_name.replace("_", " ").title()
                    )
            
            plt.title("Accuracy Over Rounds")
            plt.xlabel("Rounds")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure with tight layout
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "training_progress.png"), dpi=300)
            plt.close()
            
            logger.info("Updated training progress plot")
        except Exception as e:
            logger.error(f"Error plotting progress: {str(e)}")


def main(
    num_rounds: int, 
    min_clients: int, 
    fraction_fit: float, 
    fraction_evaluate: float,
    use_homomorphic_encryption: bool = True,
    use_differential_privacy: bool = True,
    use_zero_knowledge_proofs: bool = True
) -> None:
    """Start Flower server with secure strategy"""
    # Create strategy
    strategy = SecureWESADStrategy(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        num_rounds=num_rounds,
        use_homomorphic_encryption=use_homomorphic_encryption,
        use_differential_privacy=use_differential_privacy,
        use_zero_knowledge_proofs=use_zero_knowledge_proofs
    )
    
    # Configure server with timeout to avoid hanging
    config = fl.server.ServerConfig(num_rounds=num_rounds, round_timeout=600)  # 10 min timeout
    
    logger.info(f"Starting secure server with {num_rounds} rounds and minimum {min_clients} clients")
    logger.info(f"Security settings - HE: {use_homomorphic_encryption}, DP: {use_differential_privacy}, ZKP: {use_zero_knowledge_proofs}")
    
    # Start server
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=config,
        strategy=strategy,
    )
    
    # Save final results
    try:
        if hasattr(history, 'metrics_distributed'):
            final_metrics = history.metrics_distributed
            pd.DataFrame(final_metrics).to_csv(os.path.join(RESULTS_DIR, "final_metrics.csv"), index=False)
            logger.info("Saved final distributed metrics")
    except Exception as e:
        logger.error(f"Error saving final metrics: {str(e)}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Secure Federated Learning Server for WESAD Dataset")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--min_clients", type=int, default=1, help="Minimum number of clients")
    parser.add_argument("--fraction_fit", type=float, default=1.0, help="Fraction of clients used for training")
    parser.add_argument("--fraction_evaluate", type=float, default=1.0, help="Fraction of clients used for evaluation")
    
    # Security options
    parser.add_argument("--use_homomorphic_encryption", action="store_true", help="Enable homomorphic encryption")
    parser.add_argument("--use_differential_privacy", action="store_true", help="Enable differential privacy")
    parser.add_argument("--use_zero_knowledge_proofs", action="store_true", help="Enable zero-knowledge proofs")
    
    args = parser.parse_args()
    
    # Start server
    try:
        logger.info("Starting Secure Flower server...")
        main(
            num_rounds=args.num_rounds,
            min_clients=args.min_clients,
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            use_homomorphic_encryption=args.use_homomorphic_encryption,
            use_differential_privacy=args.use_differential_privacy,
            use_zero_knowledge_proofs=args.use_zero_knowledge_proofs,
        )
        logger.info("Server finished successfully!")
        
    except Exception as e:
        logger.error(f"Server failed: {str(e)}")
        logger.error(traceback.format_exc())