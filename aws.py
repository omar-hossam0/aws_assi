import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Neural Network with tanh activation function
class NeuralNetwork:
    def __init__(self):
        # Initialize weights randomly from [-0.5, 0.5]
        self.w1 = np.random.uniform(-0.5, 0.5, (2, 2))  # weights for hidden layer
        self.w2 = np.random.uniform(-0.5, 0.5, (2, 1))  # weights for output layer
        
        # Initialize biases
        self.b1 = 0.5   # bias for hidden layer
        self.b2 = 0.7   # bias for output layer
    
    def tanh(self, x):
        """tanh activation function"""
        return np.tanh(x)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Hidden layer: z1 = w1 * x + b1
        self.z1 = np.dot(x, self.w1) + self.b1
        # Apply tanh activation
        self.a1 = self.tanh(self.z1)
        
        # Output layer: z2 = w2 * a1 + b2
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        # Apply tanh activation
        self.a2 = self.tanh(self.z2)
        
        return self.a2

# Create and run the network
print("=" * 50)
print("Neural Network with tanh Activation Function")
print("=" * 50)

# Initialize network
nn = NeuralNetwork()

# Display network parameters
print("\nNetwork Parameters:")
print(f"Weights (Hidden Layer):\n{nn.w1}")
print(f"\nWeights (Output Layer):\n{nn.w2}")
print(f"\nBias b1 (Hidden Layer): {nn.b1}")
print(f"Bias b2 (Output Layer): {nn.b2}")

# Create sample input
x = np.array([0.5, -0.3])

# Forward pass
output = nn.forward(x)

# Print results
print("\n" + "=" * 50)
print("Network Output:")
print("=" * 50)
print(f"Input: {x}")
print(f"\nHidden Layer Output (after tanh): {nn.a1}")
print(f"\nFinal Network Output: {output}")
print("=" * 50)
