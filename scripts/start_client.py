#!/usr/bin/env python
import sys
import os
import argparse
import logging
import socket

# Add the project root to the Python path to ensure proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def check_server_connection(server_address):
    """Check if server is actually ready"""
    host, port_str = server_address.split(':')
    port = int(port_str)
    try:
        # Create a socket and attempt to connect
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)  # Set timeout to 10 seconds
        s.connect((host, port))
        s.close()
        logger.info(f"Successfully connected to server at {server_address}")
        return True
    except Exception as e:
        logger.error(f"Could not connect to server at {server_address}: {str(e)}")
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Start a federated learning client")
    parser.add_argument('--subject_id', type=str, required=True, help='Subject ID for client data')
    parser.add_argument('--window_size', type=int, required=True, help='Window size for data preprocessing')
    parser.add_argument('--subset_size', type=float, required=True, help='Subset size of data to use')
    parser.add_argument('--server_address', type=str, default='localhost:8080', help='Server address')
    parser.add_argument('--use_homomorphic_encryption', action='store_true', help='Enable homomorphic encryption')
    parser.add_argument('--use_differential_privacy', action='store_true', help='Enable differential privacy')
    parser.add_argument('--dp_epsilon', type=float, default=1.0, help='Epsilon value for differential privacy')
    parser.add_argument('--noise_multiplier', type=float, default=1.1, help='Noise multiplier for DP')
    parser.add_argument('--privacy_budget', type=float, default=5.0, help='Privacy budget for DP')
    parser.add_argument('--use_zero_knowledge_proofs', action='store_true', help='Enable zero-knowledge proofs')
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        
        logger.info(f"Starting client for subject {args.subject_id} with connection to server at {args.server_address}")
        
        # Check if the server is available before proceeding
        if not check_server_connection(args.server_address):
            logger.error("Server is not available. Exiting.")
            sys.exit(1)
        
        # Set environment variables for federated_client.py to use
        os.environ["FL_SUBJECT_ID"] = args.subject_id
        os.environ["FL_WINDOW_SIZE"] = str(args.window_size)
        os.environ["FL_SUBSET_SIZE"] = str(args.subset_size)
        os.environ["FL_SERVER_ADDRESS"] = args.server_address
        os.environ["FL_USE_HE"] = "1" if args.use_homomorphic_encryption else "0"
        os.environ["FL_USE_DP"] = "1" if args.use_differential_privacy else "0"
        os.environ["FL_DP_EPSILON"] = str(args.dp_epsilon)
        os.environ["FL_NOISE_MULTIPLIER"] = str(args.noise_multiplier)
        os.environ["FL_PRIVACY_BUDGET"] = str(args.privacy_budget)
        os.environ["FL_USE_ZKP"] = "1" if args.use_zero_knowledge_proofs else "0"
        
        # Import and call main function
        from scripts.federated_client import main
        main()
        
        logger.info(f"Client for subject {args.subject_id} completed successfully")
    except Exception as e:
        logger.error(f"Error in client execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)