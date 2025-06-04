# attacks/__init__.py
# Make attack modules importable
from attacks.membership_inference import run_membership_inference_attack
from attacks.gradient_leakage import run_gradient_leakage_attack
from attacks.model_inversion import run_model_inversion_attack

# benchmarks/__init__.py
# Make benchmarking modules importable
from benchmarks.comparative_analysis import run_comparative_analysis