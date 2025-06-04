#!/usr/bin/env python3
"""
Orchestration script for secure federated learning with WESAD dataset.

This script coordinates the launching of both server and client processes,
ensuring proper synchronization, resource management, and error handling.
It implements security features including homomorphic encryption, differential
privacy, and zero-knowledge proofs.
"""
import time
import os
import sys
import time
import logging
import argparse
import subprocess
import concurrent.futures
import socket
import signal
import contextlib
from typing import List, Dict, Optional, Tuple, Any, Set, Union, Iterator
import pandas as pd
import queue
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("federated_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PORT: int = 8080
DEFAULT_IP: str = "localhost"
DEFAULT_NUM_ROUNDS: int = 5
DEFAULT_SUBJECTS: List[str] = ["2", "3", "4", "5", "6", "7", "8", "11", "13", "14", "15", "16", "17"]
SERVER_STARTUP_TIMEOUT: int = 90  # seconds
CLIENT_EXECUTION_TIMEOUT: int = 7200  # 2 hours
SERVER_COMPLETION_TIMEOUT: int = 600  # 10 minutes
CLIENT_LAUNCH_DELAY: int = 3  # seconds
SERVER_INIT_DELAY: int = 10  # seconds
CLIENT_STARTUP_TIMEOUT: int = 30  # seconds to check if client started successfully

# Queue for monitoring server startup status
server_ready_queue: queue.Queue = queue.Queue()
# Set to track running processes for cleanup
running_processes: Set[subprocess.Popen] = set()

@contextlib.contextmanager
def process_management() -> Iterator[None]:
    """
    Context manager for proper process cleanup on exit or exception.
    Ensures all spawned processes are terminated when the main script exits.
    """
    try:
        yield
    finally:
        cleanup_processes()

def cleanup_processes() -> None:
    """
    Terminate all running processes tracked by the script.
    This ensures clean exit without orphaned processes.
    """
    for process in running_processes:
        try:
            if process.poll() is None:  # Process is still running
                logger.info(f"Terminating process with PID {process.pid}")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.terminate()
                process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError, OSError) as e:
            logger.warning(f"Error terminating process: {str(e)}")
            try:
                process.kill()
            except:
                pass

def check_port_available(host: str, port: int) -> bool:
    """
    Check if a network port is available.
    
    Args:
        host: Hostname or IP address
        port: Port number to check
        
    Returns:
        True if the port is available, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False

def setup_environment() -> Dict[str, str]:
    """
    Set up the environment variables, particularly PYTHONPATH.
    
    Returns:
        Dictionary of environment variables
    """
    env = os.environ.copy()
    # Get absolute path to project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_root)
    
    # Handle PYTHONPATH for imports
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}:{parent_dir}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = f"{project_root}:{parent_dir}"
    
    # Set unbuffered Python output
    env["PYTHONUNBUFFERED"] = "1"
    
    return env

def start_server(
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    min_clients: int = 2,
    use_he: bool = False,
    use_dp: bool = False,
    use_zkp: bool = False
) -> subprocess.Popen:
    """
    Start the federated learning server.
    
    Args:
        num_rounds: Number of federated rounds
        min_clients: Minimum number of clients required
        use_he: Enable homomorphic encryption
        use_dp: Enable differential privacy
        use_zkp: Enable zero-knowledge proofs
        
    Returns:
        Server process handle
    """
    server_cmd = [
        sys.executable, 
        "-u",  # Run Python in unbuffered mode
        "scripts/federated_server.py",
        "--num_rounds", str(num_rounds),
        "--min_clients", str(min_clients),
        "--fraction_fit", "1.0",
        "--fraction_evaluate", "1.0"
    ]
    
    # Add security flags if enabled
    if use_he:
        server_cmd.append("--use_homomorphic_encryption")
    if use_dp:
        server_cmd.append("--use_differential_privacy")
    if use_zkp:
        server_cmd.append("--use_zero_knowledge_proofs")
    
    logger.info(f"Starting server with command: {' '.join(server_cmd)}")
    
    # Set up environment
    env = setup_environment()
    
    # Start the server as a subprocess with process group
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,  # Line buffered
        universal_newlines=True,
        preexec_fn=os.setsid  # Create new process group for clean termination
    )
    
    # Track the process for cleanup
    running_processes.add(server_process)
    
    return server_process

def start_client(
    subject_id: str,
    server_address: str,
    window_size: int = 60,
    subset_size: float = 0.2,
    use_he: bool = False,
    use_dp: bool = False,
    use_zkp: bool = False,
    dp_epsilon: float = 1.0,
    noise_multiplier: float = 1.1,
    privacy_budget: float = 5.0
) -> subprocess.Popen:
    """
    Start a federated learning client for a specific subject.
    
    Args:
        subject_id: Subject ID
        server_address: Server address (host:port)
        window_size: Window size for time series data
        subset_size: Fraction of data to use
        use_he: Enable homomorphic encryption
        use_dp: Enable differential privacy
        use_zkp: Enable zero-knowledge proofs
        dp_epsilon: Differential privacy epsilon parameter
        noise_multiplier: Noise multiplier for DP
        privacy_budget: Total privacy budget
        
    Returns:
        Client process handle
    """
    client_cmd = [
        sys.executable,
        "-u",  # Run Python in unbuffered mode
        "scripts/start_client.py",
        "--subject_id", subject_id,
        "--window_size", str(window_size),
        "--subset_size", str(subset_size),
        "--server_address", server_address
    ]
    
    # Add security flags if enabled
    if use_he:
        client_cmd.append("--use_homomorphic_encryption")
    if use_dp:
        client_cmd.append("--use_differential_privacy")
        client_cmd.extend(["--dp_epsilon", str(dp_epsilon)])
        client_cmd.extend(["--noise_multiplier", str(noise_multiplier)])
        client_cmd.extend(["--privacy_budget", str(privacy_budget)])
    if use_zkp:
        client_cmd.append("--use_zero_knowledge_proofs")
    
    logger.info(f"Starting client for subject {subject_id} with command: {' '.join(client_cmd)}")
    
    # Set up environment
    env = setup_environment()
    
    # Start the client as a subprocess with process group
    client_process = subprocess.Popen(
        client_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,  # Line buffered
        universal_newlines=True,
        preexec_fn=os.setsid  # Create new process group for clean termination
    )
    
    # Track the process for cleanup
    running_processes.add(client_process)
    
    # Verify client started successfully
    if client_process.poll() is not None:
        logger.error(f"Client {subject_id} failed to start")
        return None
    
    # Wait a moment to check if client crashes immediately
    time.sleep(1)
    if client_process.poll() is not None:
        logger.error(f"Client {subject_id} crashed after starting")
        return None
    
    return client_process

def monitor_client(process: subprocess.Popen, client_name: str) -> bool:
    """
    Monitor a client process and log its output.
    
    Args:
        process: Client process to monitor
        client_name: Name identifier for client logs
        
    Returns:
        True if client completed successfully, False if errors occurred
    """
    success = True
    
    try:
        for line in iter(process.stdout.readline, ''):
            logger.info(f"{client_name}: {line.strip()}")
            
            # Check for error conditions
            if any(error_text in line for error_text in ["Error", "ERROR", "error", "Exception", "Failed", "failed"]):
                logger.error(f"{client_name} error detected: {line.strip()}")
                success = False
    except Exception as e:
        logger.error(f"Error monitoring {client_name}: {str(e)}")
        logger.error(traceback.format_exc())
        success = False
    
    return success

def monitor_server(process: subprocess.Popen) -> None:
    """
    Monitor the server process and detect when it's ready for connections.
    
    Args:
        process: Server process to monitor
    """
    server_ready = False
    
    try:
        for line in iter(process.stdout.readline, ''):
            logger.info(f"Server: {line.strip()}")
            
            # Update this pattern to match your server's actual ready message
            if "gRPC server running" in line or "Flower server started" in line or "Server started" in line:
                server_ready = True
                server_ready_queue.put(True)
                logger.info("Server is ready for connections")
            
            # Check for error conditions
            if any(error_text in line for error_text in ["Error", "ERROR", "error", "Exception", "Failed", "failed"]):
                logger.error(f"Server error detected: {line.strip()}")
                if not server_ready and server_ready_queue.empty():
                    server_ready_queue.put(False)
                    return
        
        if not server_ready and server_ready_queue.empty():
            logger.error("Server process ended without becoming ready")
            server_ready_queue.put(False)
            
    except Exception as e:
        logger.error(f"Error monitoring server: {str(e)}")
        logger.error(traceback.format_exc())
        if server_ready_queue.empty():
            server_ready_queue.put(False)

def ensure_directories() -> None:
    """Create necessary directories for results and logs."""
    directories = ["results", "server_results", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def wait_for_server_ready() -> bool:
    """
    Wait for server to be ready using polling mechanism.
    
    Returns:
        True if server is ready, False if not
    """
    logger.info(f"Waiting up to {SERVER_STARTUP_TIMEOUT} seconds for server to initialize...")
    try:
        server_ready = server_ready_queue.get(timeout=SERVER_STARTUP_TIMEOUT)
        if not server_ready:
            logger.error("Server failed to start properly")
            return False
        logger.info("Server is ready for connections")
        return True
    except queue.Empty:
        logger.error("Timed out waiting for server to initialize")
        return False

def run_federated_learning(
    subjects: List[str],
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    window_size: int = 60,
    subset_size: float = 0.2,
    server_address: str = f"{DEFAULT_IP}:{DEFAULT_PORT}",
    use_he: bool = False,
    use_dp: bool = False,
    use_zkp: bool = False,
    dp_epsilon: float = 1.0,
    noise_multiplier: float = 1.1,
    privacy_budget: float = 5.0
) -> None:
    with process_management():
        try:
            # Parse server address
            if ":" in server_address:
                host, port_str = server_address.split(":")
                port = int(port_str)
            else:
                host, port = DEFAULT_IP, DEFAULT_PORT
                server_address = f"{host}:{port}"
                
            # Check if server port is available
            if not check_port_available(host, port):
                logger.error(f"Port {port} is already in use. Please stop any existing servers.")
                return
            
            # Create necessary directories
            ensure_directories()
            
            # Start the server
            min_clients = max(1, min(len(subjects) // 2, 2))
            logger.info(f"Setting minimum required clients to {min_clients}")
            
            server_process = start_server(
                num_rounds=num_rounds,
                min_clients=min_clients,
                use_he=use_he,
                use_dp=use_dp,
                use_zkp=use_zkp
            )
            
            # Start monitoring the server in a separate thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                server_monitor = executor.submit(monitor_server, server_process)
                
                # Wait for server to be ready using improved waiting method
                if not wait_for_server_ready():
                    return
                
                # Start first client immediately to provide initial parameters
                logger.info("Server is ready. Starting first client to provide initial parameters...")
                first_subject = subjects[0]
                first_client = start_client(
                    subject_id=first_subject,
                    server_address=server_address,
                    window_size=window_size,
                    subset_size=subset_size,
                    use_he=use_he,
                    use_dp=use_dp,
                    use_zkp=use_zkp,
                    dp_epsilon=dp_epsilon,
                    noise_multiplier=noise_multiplier,
                    privacy_budget=privacy_budget
                )
                
                # Verify client started successfully
                if first_client is None:
                    logger.error("First client failed to start. Aborting.")
                    return
                
                # Monitor the first client
                first_client_monitor = executor.submit(
                    monitor_client, 
                    first_client, 
                    f"Client-{first_subject}"
                )
                
                # Create the client process list and add the first client
                client_processes = [(first_client, first_subject)]
                client_monitors = [first_client_monitor]
                
                # Wait for first client to establish connection
                logger.info(f"Waiting {SERVER_INIT_DELAY} seconds for first client to establish connection...")
                time.sleep(SERVER_INIT_DELAY)
                
                # Check if first client is still running
                if first_client.poll() is not None:
                    logger.error("First client terminated unexpectedly. Aborting.")
                    return
                    
                logger.info("Starting remaining clients...")
                
                # Use a smaller batch of subjects for testing if needed
                active_subjects = subjects[:2] if os.environ.get("FL_DEBUG") else subjects
                
                # Track successful client launches
                successful_client_count = 1  # First client already started
                
                # Start the remaining clients (skipping the first one)
                for subject_id in active_subjects[1:]:
                    client_process = start_client(
                        subject_id=subject_id,
                        server_address=server_address,
                        window_size=window_size,
                        subset_size=subset_size,
                        use_he=use_he,
                        use_dp=use_dp,
                        use_zkp=use_zkp,
                        dp_epsilon=dp_epsilon,
                        noise_multiplier=noise_multiplier,
                        privacy_budget=privacy_budget
                    )
                    
                    # Verify client started successfully
                    if client_process is None:
                        logger.error(f"Client {subject_id} failed to start. Continuing with other clients.")
                        continue
                    
                    successful_client_count += 1
                    client_processes.append((client_process, subject_id))
                    
                    # Monitor each client in a separate thread
                    client_monitor = executor.submit(
                        monitor_client, 
                        client_process, 
                        f"Client-{subject_id}"
                    )
                    client_monitors.append(client_monitor)
                    
                    # Add a small delay between client launches
                    time.sleep(CLIENT_LAUNCH_DELAY)
                
                # Verify minimum number of clients are running
                if successful_client_count < min_clients:
                    logger.error(f"Failed to start minimum required clients ({successful_client_count}/{min_clients}). Aborting.")
                    return
                
                logger.info(f"Successfully started {successful_client_count} clients")
                
                # Wait for all clients to complete
                for client_process, subject_id in client_processes:
                    try:
                        ret_code = client_process.wait(timeout=CLIENT_EXECUTION_TIMEOUT)
                        logger.info(f"Client for subject {subject_id} completed with return code {ret_code}")
                        running_processes.discard(client_process)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Client for subject {subject_id} timed out, terminating")
                        try:
                            os.killpg(os.getpgid(client_process.pid), signal.SIGTERM)
                            client_process.terminate()
                            running_processes.discard(client_process)
                        except Exception as e:
                            logger.error(f"Error terminating client {subject_id}: {str(e)}")
                
                # Wait for server to complete
                try:
                    ret_code = server_process.wait(timeout=SERVER_COMPLETION_TIMEOUT)
                    logger.info(f"Server completed with return code {ret_code}")
                    running_processes.discard(server_process)
                except subprocess.TimeoutExpired:
                    logger.warning("Server timed out, terminating")
                    try:
                        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                        server_process.terminate()
                        running_processes.discard(server_process)
                    except Exception as e:
                        logger.error(f"Error terminating server: {str(e)}")
            
            # Collect and analyze results
            analyze_results(subjects, use_he, use_dp, use_zkp, dp_epsilon, noise_multiplier)
            
        except Exception as e:
            logger.error(f"Error in federated learning: {str(e)}")
            logger.error(traceback.format_exc())

def analyze_results(
    subjects: List[str],
    use_he: bool = False,
    use_dp: bool = False, 
    use_zkp: bool = False,
    dp_epsilon: float = 1.0,
    noise_multiplier: float = 1.1
) -> None:
    """
    Analyze the results of federated learning.
    
    Args:
        subjects: List of subject IDs
        use_he: Homomorphic encryption was used
        use_dp: Differential privacy was used
        use_zkp: Zero-knowledge proofs were used
        dp_epsilon: Epsilon value for differential privacy
        noise_multiplier: Noise multiplier for differential privacy
    """
    try:
        logger.info("Analyzing federated learning results")
        
        # Initialize the results_summary dictionary first
        results_summary = {
            "global_metrics": {},
            "client_metrics": {},
            "security_settings": {
                "homomorphic_encryption": use_he,
                "differential_privacy": use_dp,
                "zero_knowledge_proofs": use_zkp
            }
        }
        
        # Load global metrics
        global_metrics_path = os.path.join("server_results", "global_metrics.csv")
        if os.path.exists(global_metrics_path):
            global_metrics = pd.read_csv(global_metrics_path)
            
            # Print summary of global metrics
            logger.info("Global metrics summary:")
            for column in global_metrics.columns:
                if column != "round":
                    values = global_metrics[column].dropna()
                    if len(values) > 0:
                        mean_val = float(values.mean())
                        final_val = float(values.iloc[-1])
                        logger.info(f"  {column}: mean={mean_val:.4f}, final={final_val:.4f}")
                        results_summary["global_metrics"][column] = {
                            "mean": mean_val,
                            "final": final_val
                        }
        
        # Load client metrics
        client_metrics = []
        for subject in subjects:
            client_metric_path = os.path.join("results", f"S{subject}_metrics.csv")
            if os.path.exists(client_metric_path):
                metrics = pd.read_csv(client_metric_path)
                metrics["subject"] = f"S{subject}"
                client_metrics.append(metrics)
        
        if client_metrics:
            all_client_metrics = pd.concat(client_metrics, ignore_index=True)
            
            # Print summary of client metrics
            logger.info("Client metrics summary:")
            for column in ["accuracy", "f1_score", "precision", "recall"]:
                if column in all_client_metrics.columns:
                    values = all_client_metrics[column].dropna()
                    if len(values) > 0:
                        mean_val = float(values.mean())
                        min_val = float(values.min())
                        max_val = float(values.max())
                        logger.info(f"  {column}: mean={mean_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
                        results_summary["client_metrics"][column] = {
                            "mean": mean_val,
                            "min": min_val,
                            "max": max_val
                        }
            
            # Save combined client metrics
            all_client_metrics.to_csv("results/all_client_metrics.csv", index=False)
        
        # Create security report - Fix the file handle issue by keeping within a single with block
        with open("results/security_report.txt", "w") as f:
            f.write("Secure Federated Learning Report\n")
            f.write("==============================\n\n")
            
            f.write("Security Settings:\n")
            for key, value in results_summary["security_settings"].items():
                f.write(f"  {key}: {'Enabled' if value else 'Disabled'}\n")
            
            f.write("\nPerformance Impact:\n")
            if os.path.exists("server_results/training_progress.png"):
                f.write("  See training_progress.png for convergence visualization\n")
            
            if use_dp:
                f.write("\nPrivacy Analysis:\n")
                f.write("  Using differential privacy protects against membership inference attacks\n")
                f.write("  and prevents leakage of private training data through gradients.\n")
                f.write(f"  Privacy parameters: epsilon={dp_epsilon}, noise multiplier={noise_multiplier}\n")
            
            if use_he:
                f.write("\nConfidentiality Analysis:\n")
                f.write("  Homomorphic encryption ensures model updates remain encrypted during transmission\n")
                f.write("  and aggregation, providing strong confidentiality guarantees.\n")
            
            if use_zkp:
                f.write("\nAuthenticity Analysis:\n")
                f.write("  Zero-knowledge proofs verify client authenticity without revealing secrets,\n")
                f.write("  protecting against malicious participants and model poisoning attacks.\n")
            
            # Write summary statistics
            f.write("\nMetrics Summary:\n")
            f.write("----------------\n")
            
            if "global_metrics" in results_summary and results_summary["global_metrics"]:
                f.write("\nGlobal Metrics:\n")
                for metric, values in results_summary["global_metrics"].items():
                    f.write(f"  {metric}: mean={values['mean']:.4f}, final={values['final']:.4f}\n")
                    
            if "client_metrics" in results_summary and results_summary["client_metrics"]:
                f.write("\nClient Metrics:\n")
                for metric, values in results_summary["client_metrics"].items():
                    f.write(f"  {metric}: mean={values['mean']:.4f}, min={values['min']:.4f}, max={values['max']:.4f}\n")
        
        # Save results summary as JSON for further analysis
        import json
        with open("results/results_summary.json", "w") as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info("Analysis completed and reports generated")
        
    except Exception as e:
        logger.error(f"Error analyzing results: {str(e)}")
        logger.error(traceback.format_exc())

def main() -> None:
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Secure Federated Learning for WESAD Dataset")
    
    # Basic federated learning parameters
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS, 
                        help="List of subject IDs to include")
    parser.add_argument("--num_rounds", type=int, default=DEFAULT_NUM_ROUNDS,
                        help="Number of federated rounds")
    parser.add_argument("--window_size", type=int, default=60,
                        help="Window size for time series data")
    parser.add_argument("--subset_size", type=float, default=0.2,
                        help="Fraction of data to use (0.0-1.0)")
    parser.add_argument("--server_address", type=str, default=f"{DEFAULT_IP}:{DEFAULT_PORT}",
                        help="Server address (host:port)")
    
    # Security parameters
    parser.add_argument("--use_he", action="store_true",
                        help="Enable homomorphic encryption")
    parser.add_argument("--use_dp", action="store_true", 
                        help="Enable differential privacy")
    parser.add_argument("--use_zkp", action="store_true",
                        help="Enable zero-knowledge proofs")
    parser.add_argument("--dp_epsilon", type=float, default=1.0,
                        help="Differential privacy epsilon parameter")
    parser.add_argument("--noise_multiplier", type=float, default=1.1,
                        help="Noise multiplier for differential privacy")
    parser.add_argument("--privacy_budget", type=float, default=5.0,
                        help="Total privacy budget for differential privacy")
                        
    # Debug mode for testing with fewer subjects
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode with fewer clients")
    
    args = parser.parse_args()
    
    # Set debug environment variable if needed
    if args.debug:
        os.environ["FL_DEBUG"] = "1"
    
    logger.info("Starting secure federated learning")
    logger.info(f"Security settings: HE={args.use_he}, DP={args.use_dp}, ZKP={args.use_zkp}")
    
    run_federated_learning(
        subjects=args.subjects,
        num_rounds=args.num_rounds,
        window_size=args.window_size,
        subset_size=args.subset_size,
        server_address=args.server_address,
        use_he=args.use_he,
        use_dp=args.use_dp,
        use_zkp=args.use_zkp,
        dp_epsilon=args.dp_epsilon,
        noise_multiplier=args.noise_multiplier,
        privacy_budget=args.privacy_budget
    )

if __name__ == "__main__":
    main()