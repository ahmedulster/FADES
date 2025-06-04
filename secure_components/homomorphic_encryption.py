# secure_components/homomorphic_encryption.py
import math
import numpy as np
import logging
import secrets
import time 
from typing import List, Dict, Any, Tuple, Optional


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PaillierEncryption:
    """
    Implementation of Paillier cryptosystem for homomorphic encryption
    
    This is a simplified version for educational purposes. In production,
    use established libraries like python-paillier or TenSEAL.
    """
    
    def __init__(self, key_size: int = 1024):
        """
        Initialize Paillier cryptosystem with given key size
        
        Args:
            key_size: Size of the keys in bits
        """
        self.key_size = key_size
        self.generate_keys()
        logger.info(f"Initialized Paillier encryption with {key_size}-bit keys")
    
    def generate_keys(self):
        """Generate public and private keys"""
        # This is a simplified implementation
        # In practice, use a proper cryptographic library
        
        # Generate two large primes p and q
        self.p = self._generate_prime(self.key_size // 2)
        self.q = self._generate_prime(self.key_size // 2)
        
        # Compute n = p * q
        self.n = self.p * self.q
        
        # Compute lambda = lcm(p-1, q-1)
        self.lambda_val = self._lcm(self.p - 1, self.q - 1)
        
        # Choose a random g in Z*_n^2
        self.g = self._random_coprime(self.n * self.n)
        
        # Compute mu = (L(g^lambda mod n^2))^(-1) mod n
        g_lambda = pow(self.g, self.lambda_val, self.n * self.n)
        L_g_lambda = self._L_function(g_lambda)
        self.mu = pow(L_g_lambda, -1, self.n)
        
        logger.info("Generated Paillier keys")
    
    def _generate_prime(self, bits: int) -> int:
        """
        Generate a prime number with the given number of bits
        
        In practice, use a proper cryptographic library
        """
        # This is a very simplified implementation for demonstration
        # In practice, use a proper primality test and generation algorithm
        
        # For educational purposes only - not secure!
        # Start with a random odd number
        num = secrets.randbits(bits) | 1
        
        # Simplistic primality test (not secure or efficient)
        while not all(num % i != 0 for i in range(3, min(1000, int(math.sqrt(num)) + 1), 2)):
            num += 2
        
        return num
    
    def _random_coprime(self, n: int) -> int:
        """Generate a random number coprime to n"""
        while True:
            g = secrets.randbelow(n)
            if self._gcd(g, n) == 1:
                return g
    
    def _gcd(self, a: int, b: int) -> int:
        """Compute greatest common divisor of a and b"""
        while b:
            a, b = b, a % b
        return a
    
    def _lcm(self, a: int, b: int) -> int:
        """Compute least common multiple of a and b"""
        return a * b // self._gcd(a, b)
    
    def _L_function(self, x: int) -> int:
        """
        L function defined as L(x) = (x - 1) / n
        """
        return (x - 1) // self.n
    
    def encrypt(self, m) -> int:
        """
        Encrypt a message m
        
        Args:
            m: The message to encrypt (must be < n)
            
        Returns:
            Encrypted ciphertext
        """
        # Convert numpy integers to Python int
        if isinstance(m, np.integer):
            m = int(m)
        
        if m >= self.n:
            raise ValueError(f"Message {m} is too large for n = {self.n}")
        
        # Choose a random r in Z*_n
        r = self._random_coprime(self.n)
        
        # Compute ciphertext c = g^m * r^n mod n^2
        n_squared = self.n * self.n
        g_m = pow(self.g, m, n_squared)
        r_n = pow(r, self.n, n_squared)
        c = (g_m * r_n) % n_squared
        
        return c
    
    def decrypt(self, c: int) -> int:
        """
        Decrypt a ciphertext c
        
        Args:
            c: The ciphertext to decrypt
            
        Returns:
            Original message
        """
        # Convert numpy integers to Python int
        if isinstance(c, np.integer):
            c = int(c)
            
        # Convert from string if needed
        if isinstance(c, str):
            c = int(c)
            
        n_squared = self.n * self.n
        
        # Compute c^lambda mod n^2
        c_lambda = pow(c, self.lambda_val, n_squared)
        
        # Apply L function
        L_c_lambda = self._L_function(c_lambda)
        
        # Compute m = L(c^lambda mod n^2) * mu mod n
        m = (L_c_lambda * self.mu) % self.n
        
        return m
    
    def add(self, c1: int, c2: int) -> int:
        """
        Homomorphic addition: E(m1) * E(m2) = E(m1 + m2)
        
        Args:
            c1: First ciphertext
            c2: Second ciphertext
            
        Returns:
            Ciphertext representing the sum
        """
        # Convert numpy integers to Python int
        if isinstance(c1, np.integer):
            c1 = int(c1)
        if isinstance(c2, np.integer):
            c2 = int(c2)
        
        # Convert from string if needed
        if isinstance(c1, str):
            c1 = int(c1)
        if isinstance(c2, str):
            c2 = int(c2)
            
        return (c1 * c2) % (self.n * self.n)
    
    def multiply_plain(self, c: int, m: int) -> int:
        """
        Homomorphic multiplication by a constant: E(m1)^m2 = E(m1 * m2)
        
        Args:
            c: Ciphertext
            m: Plaintext constant
            
        Returns:
            Ciphertext representing the product
        """
        # Convert numpy integers to Python int
        if isinstance(c, np.integer):
            c = int(c)
        if isinstance(m, np.integer):
            m = int(m)
        
        # Convert from string if needed
        if isinstance(c, str):
            c = int(c)
        if isinstance(m, str):
            m = int(m)
            
        if m < 0:
            # For negative m, we need to compute the modular multiplicative inverse
            c_inv = pow(c, -1, self.n * self.n)
            return pow(c_inv, abs(m), self.n * self.n)
        else:
            return pow(c, m, self.n * self.n)
    
    def encrypt_tensor(self, tensor):
        """
        Encrypt a tensor value by value using lightweight approach.
        Only encrypts significant values to reduce computational overhead.
        
        Args:
            tensor: TensorFlow tensor or numpy array to encrypt
            
        Returns:
            Dictionary with encrypted values and metadata
        """
        # Convert to numpy if it's a TensorFlow tensor
        if hasattr(tensor, 'numpy'):
            np_tensor = tensor.numpy()
        else:
            np_tensor = tensor
        
        # Flatten the tensor
        original_shape = np_tensor.shape
        flattened = np_tensor.flatten()
        
        # Calculate magnitude thresholds - only encrypt values above threshold
        abs_values = np.abs(flattened)
        # Use standard deviation as a dynamic threshold
        magnitude_threshold = np.std(abs_values) * 0.1
        
        # Find significant indices (values above threshold)
        significant_indices = np.where(abs_values > magnitude_threshold)[0]
        
        # If too few significant values, adjust threshold
        if len(significant_indices) < 10:
            # Find top 10% values by magnitude
            num_values = max(int(len(flattened) * 0.1), 10)
            significant_indices = np.argsort(abs_values)[-num_values:]
        
        # Cap the number of encrypted values to prevent excessive computation
        max_encrypted_values = min(10000, len(significant_indices))
        if len(significant_indices) > max_encrypted_values:
            # Select the most significant values
            idx_by_importance = np.argsort(abs_values[significant_indices])[-max_encrypted_values:]
            significant_indices = significant_indices[idx_by_importance]
        
        # Scale values to integers (Paillier works with integers)
        scale_factor = 1000.0
        scaled = (flattened[significant_indices] * scale_factor).astype(int)
        
        # Start timer for encryption
        start_time = time.time()
        
        # Encrypt only significant values
        encrypted_values = []
        for value in scaled:
            # Store encrypted values as strings to avoid overflow
            encrypted_value = self.encrypt(value)
            encrypted_values.append(str(encrypted_value))
        
        # Calculate encryption time
        encryption_time = time.time() - start_time
        
        # Calculate sparsity for logging
        sparsity = 1.0 - (len(significant_indices) / len(flattened))
        logger.info(f"Encrypted tensor with {sparsity:.2%} sparsity ({len(significant_indices)} of {len(flattened)} values)")
        logger.info(f"Encryption completed in {encryption_time:.2f} seconds")
        
        # Store non-encrypted placeholder values for all indices
        # In a real system, you would store zeros or small noise
        placeholder_values = np.zeros(len(flattened))
        
        # Store indices, encrypted values, and metadata
        return {
            "encrypted": True,
            "original_shape": original_shape,
            "scale_factor": scale_factor,
            "indices": significant_indices.tolist(),
            "encrypted_values": encrypted_values,
            "non_encrypted": np_tensor,  # For testing only! Remove in production
            "placeholder_values": placeholder_values,
            "sparsity": sparsity,
            "encryption_time": encryption_time
        }

    def decrypt_tensor(self, encrypted_tensor):
        """
        Decrypt an encrypted tensor.
        
        Args:
            encrypted_tensor: Dictionary with encrypted tensor data
            
        Returns:
            NumPy array with decrypted values
        """
        if not isinstance(encrypted_tensor, dict) or not encrypted_tensor.get("encrypted", False):
            return encrypted_tensor
        
        # Extract metadata
        original_shape = encrypted_tensor["original_shape"]
        scale_factor = encrypted_tensor["scale_factor"]
        indices = encrypted_tensor["indices"]
        encrypted_values = encrypted_tensor["encrypted_values"]
        
        # Start timer for decryption
        start_time = time.time()
        
        # Initialize array with placeholder values or zeros
        if "placeholder_values" in encrypted_tensor:
            decrypted_flat = encrypted_tensor["placeholder_values"].copy()
        else:
            decrypted_flat = np.zeros(np.prod(original_shape), dtype=float)
        
        # Decrypt each encrypted value and place at correct index
        for i, idx in enumerate(indices):
            # Convert string back to int for decryption
            encrypted_value = encrypted_values[i]
            if isinstance(encrypted_value, str):
                encrypted_value = int(encrypted_value)
            
            decrypted_int = self.decrypt(encrypted_value)
            decrypted_flat[idx] = decrypted_int / scale_factor
        
        # Calculate decryption time
        decryption_time = time.time() - start_time
        logger.info(f"Decryption completed in {decryption_time:.2f} seconds")
        
        # Reshape to original shape
        decrypted = decrypted_flat.reshape(original_shape)
        
        return decrypted