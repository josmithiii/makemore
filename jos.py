import torch
import torch.nn as nn

class DelayLine(nn.Module):
    def __init__(self, L, B, D):
        """
        Initialize the DelayLine module.
        
        Parameters:
        - L: The length of the delay in calls.
        - B: The batch size of the input and output tensors.
        - D: The dimensionality of the data in the input and output tensors.
        """
        super(DelayLine, self).__init__()
        self.L = L
        self.B = B
        self.D = D
        self.buffer = torch.zeros((L, B, D))  # Initialize the circular buffer
        self.index = 0  # Current index for inserting the next tensor
        self.call_count = 0  # Count how many times forward() has been called

    def forward(self, x):
        """
        Forward pass of the DelayLine.

        Parameters:
        - x: The input tensor of shape (B, D).

        Returns:
        - The output tensor of shape (B, D). It will be zeros for the first L-1 calls,
          and starting from the Lth call, it will be the input tensor from L calls ago.
        """
        # Check if the input tensor has the expected shape
        assert x.shape == (self.B, self.D), "Input tensor does not match expected shape."

        # Determine the output
        if self.call_count < self.L:
            # For the first L-1 calls, output zeros
            output = torch.zeros_like(x)
        else:
            # Starting from the Lth call, output the tensor from L calls ago
            # The output should be retrieved before updating the buffer
            output = self.buffer[self.index].clone()  # Use .clone() to ensure a true copy is made

        # Insert the current input tensor into the buffer at the current index
        self.buffer[self.index] = x

        # Update the index, wrapping around to create a circular buffer
        self.index = (self.index + 1) % self.L

        # Increment the call count
        self.call_count += 1

        return output

# Example usage
L = 3  # Delay length
B = 2  # Batch size
D = 4  # Dimensionality of the data

delay_line = DelayLine(L, B, D)

# Simulate multiple calls to forward() to demonstrate the delay effect
for i in range(5):
    x = torch.randn(B, D)  # Random input tensor
    print(f"Call {i+1}, Input:\n{x}")
    output = delay_line(x)
    print(f"Call {i+1}, Output:\n{output}\n")

