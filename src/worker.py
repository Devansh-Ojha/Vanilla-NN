import sys
import os
import time
import grpc
import numpy as np

# --- IMPORT FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ------------------

import proto.parameter_server_pb2 as pb2
import proto.parameter_server_pb2_grpc as pb2_grpc
from src.utils import serialize_tensor, deserialize_tensor

def run():
    # 1. Connect to the Server
    # "localhost:50051" works because we are running locally for now.
    # When we use Docker later, this will change to the container name.
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb2_grpc.ParameterServerStub(channel)
    
    worker_id = "worker_1"
    print(f"[{worker_id}] Connecting to server...")

    # 2. Simulate Training Loop
    for step in range(5):
        print(f"\n--- Step {step} ---")
        
        # A. Simulate calculating gradients (Random noise for now)
        # In real life, this is: loss.backward()
        gradients = np.random.randn(10).astype(np.float32)
        print(f"[{worker_id}] Computed Gradients: {gradients[:3]}...") # Print first 3
        
        # B. Send to Server
        request = pb2.GradientRequest(
            worker_id=worker_id,
            step=step,
            gradient_data=serialize_tensor(gradients),
            shape=list(gradients.shape)
        )
        
        # C. Wait for response (Synchronous)
        response = stub.PushGradients(request)
        
        # D. Update local weights
        new_weights = deserialize_tensor(response.weight_data)
        print(f"[{worker_id}] Received New Weights: {new_weights}")
        
        # Sleep to simulate computation time
        time.sleep(1)

if __name__ == '__main__':
    run()