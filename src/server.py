import sys
import os
import time
from concurrent import futures
import grpc
import numpy as np

# --- IMPORT FIX ---
# This allows python to find the 'proto' folder even if you run this script from inside 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ------------------

import proto.parameter_server_pb2 as pb2
import proto.parameter_server_pb2_grpc as pb2_grpc
from src.utils import serialize_tensor, deserialize_tensor

class ParameterServer(pb2_grpc.ParameterServerServicer):
    def __init__(self):
        # Initialize weights: A simple array of 10 floats (initialized to 0)
        self.weights = np.zeros(10, dtype=np.float32)
        print(f"Server initialized. Weights: {self.weights}")

    def PushGradients(self, request, context):
        # Deserialize the incoming gradient bytes back to numpy
        gradients = deserialize_tensor(request.gradient_data)
        
        print(f"[Server] Received gradients from {request.worker_id}")
        
        # --- THE MATH ---
        # Simple Gradient Descent: NewWeights = OldWeights - (LearningRate * Gradients)
        learning_rate = 0.1
        self.weights = self.weights - (learning_rate * gradients)
        
        print(f"[Server] Updated Weights: {self.weights}")

        # Send the updated weights back to the worker
        return pb2.WeightResponse(
            weight_data=serialize_tensor(self.weights),
            shape=list(self.weights.shape)
        )

def serve():
    # Start the gRPC server with 10 worker threads
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_ParameterServerServicer_to_server(ParameterServer(), server)
    
    # Listen on port 50051
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Parameter Server is running on port 50051...")
    
    # Keep the script running
    server.wait_for_termination()

if __name__ == '__main__':
    serve()