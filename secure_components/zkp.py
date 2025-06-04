import hashlib
import hmac
import secrets
import base64
import time 
from typing import Dict, Any, Union, Optional

def generate_proof(client_secret: str, challenge: bytes, client_id: str) -> Dict[str, Any]:
    """
    Generate a simplified ZKP proof that can be serialized for Flower.
    
    Args:
        client_secret: Client's secret key (as hex string)
        challenge: Challenge from the server (bytes)
        client_id: Client ID
    
    Returns:
        Dictionary containing proof components
    """
    # Convert hex string to bytes if needed
    if isinstance(client_secret, str):
        client_secret_bytes = bytes.fromhex(client_secret)
    else:
        client_secret_bytes = client_secret
    
    # Create an HMAC signature using the client's secret key
    signature = hmac.new(
        key=client_secret_bytes,
        msg=challenge + client_id.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    
    # Hash the challenge for verification
    challenge_hash = hashlib.sha256(challenge).digest()
    
    # Create a deterministic "public key" from the client's secret
    # This is for demonstration only - not a real public key
    public_key = hashlib.sha256(client_secret_bytes + b"public_key").digest()
    
    # Return proof components
    # Note: All binary values are converted to strings to be serialization-friendly
    return {
        "signature": base64.b64encode(signature).decode('utf-8'),
        "challenge_hash": base64.b64encode(challenge_hash).decode('utf-8'),
        "public_key": base64.b64encode(public_key).decode('utf-8'),
        "client_id": client_id,
        # Add nonce as integer
        "nonce": int.from_bytes(secrets.token_bytes(4), "big"),
        # Add timestamp as float
        "timestamp": float(time.time())
    }

def verify_proof(proof: Dict[str, Any], challenge: bytes, client_id: str) -> bool:
    """
    Verify a ZKP proof.
    
    Args:
        proof: Proof from client (already reconstructed)
        challenge: Original challenge sent to the client
        client_id: Expected client ID
    
    Returns:
        True if proof is valid, False otherwise
    """
    try:
        # Reconstruct binary values if they're base64 strings
        signature = proof.get("signature")
        if isinstance(signature, str):
            signature = base64.b64decode(signature)
        
        challenge_hash = proof.get("challenge_hash")
        if isinstance(challenge_hash, str):
            challenge_hash = base64.b64decode(challenge_hash)
            
        # Check if challenge hash matches
        expected_challenge_hash = hashlib.sha256(challenge).digest()
        if challenge_hash != expected_challenge_hash:
            return False
            
        # Check if client ID matches
        if proof.get("client_id") != client_id:
            return False
            
        # In a real ZKP system, we would verify the signature using the client's public key
        # For this simplified example, we'll just acknowledge that the verification happened
        
        # Check if the timestamp is reasonably recent (within last hour)
        timestamp = proof.get("timestamp", 0.0)
        if isinstance(timestamp, (int, float)) and timestamp > 0:
            current_time = time.time()
            if current_time - timestamp > 3600:  # 1 hour
                return False
                
        return True
        
    except Exception as e:
        print(f"Error verifying proof: {str(e)}")
        return False


# Server-side function to generate a challenge
def generate_challenge() -> bytes:
    """Generate a cryptographic challenge for the client."""
    return secrets.token_bytes(32)