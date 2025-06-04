#!/usr/bin/env python
import sys
import os
import inspect

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from scripts.federated_client import main
    
    # Get the function signature
    signature = inspect.signature(main)
    
    print(f"Function signature for main():")
    print(f"{signature}")
    
    # Get parameters
    for param_name, param in signature.parameters.items():
        default = param.default if param.default is not inspect.Parameter.empty else "No default"
        print(f"Parameter: {param_name}, Default: {default}, Kind: {param.kind}")
        
except ImportError:
    print("Could not import main from federated_client.py")
    import traceback
    print(traceback.format_exc())
except Exception as e:
    print(f"Error inspecting function: {str(e)}")
    import traceback
    print(traceback.format_exc())