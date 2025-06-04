# Secure components package for federated learning
from secure_components.homomorphic_encryption import PaillierEncryption
from secure_components.differential_privacy import add_noise, compute_epsilon
from secure_components.zkp import generate_proof, verify_proof

__all__ = [
    'PaillierEncryption',
    'add_noise',
    'compute_epsilon',
    'generate_proof',
    'verify_proof'
]