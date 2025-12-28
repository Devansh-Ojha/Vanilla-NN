import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x)) 

def sigmoid_derivative(x):
    return x * (1-x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))

        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))

    def forward(self, inputs):
        # Hidden Layer calculations
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden)
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
    
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.predicted_output = sigmoid(self.output_layer_input)
        
        return self.predicted_output

    def train(self, inputs, expected_output, learning_rate):
        self.forward(inputs)
       
        error = expected_output - self.predicted_output
        
        d_predicted_output = error * sigmoid_derivative(self.predicted_output)
        
        error_hidden_layer = d_predicted_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(self.hidden_layer_output)
        self.weights_hidden_output += self.hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        self.weights_input_hidden += inputs.T.dot(d_hidden_layer) * learning_rate

    def save(self, filename):
        # Save weights to a .npz file (NumPy Zip)
        np.savez(filename, 
                 w1=self.weights_input_hidden, 
                 w2=self.weights_hidden_output)
        print(f"Brain saved to {filename}")

    def load(self, filename):
        # Load weights from the file
        data = np.load(filename)
        self.weights_input_hidden = data['w1']
        self.weights_hidden_output = data['w2']
        print(f"Brain loaded from {filename}")

if __name__ == "__main__":

    inputs = np.array([[0,0], [0,1], [1,0], [1,1]])

    expected_output = np.array([[0], [1], [1], [0]])


    nn = NeuralNetwork(2, 2, 1)

    print("--- Before Training (Random Guessing) ---")
    print(nn.forward(inputs))
    print("\nTraining...")
    epochs = 10000
    for i in range(epochs):
        nn.train(inputs, expected_output, learning_rate=0.1)
        if (i % 1000) == 0:
            loss = np.mean(np.square(expected_output - nn.predicted_output))
            print(f"Epoch {i}, Loss: {loss:.5f}")

    print("\n--- After Training (Results) ---")
    print(nn.forward(inputs))
    nn.save("brain.npz")

    