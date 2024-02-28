import torch
import torch.nn as nn

class DelayLine(nn.Module):
    def __init__(self, L, D):
        super(DelayLine, self).__init__()
        self.L = L
        self.D = D
        self.buffer = [] # list of input buffers
        self.index = 0

    def forward(self, x):
        B, D = x.shape
        assert D == self.D, "Input tensor dimensionality does not match expected dimension."

        self.buffer.append(x.clone().detach()) # detached copy of x added to our delay-line list
        if len(self.buffer) > self.L:
            # If the buffer size exceeds the delay length, pop the earliest tensor
            output = self.buffer.pop(0)
        else:
            # Return zeros for the first L-1 calls
            output = torch.zeros_like(x)

        return output

class DelayLine0(nn.Module):
    def __init__(self, L, D):
        """
        Initialize the DelayLine module.
        
        Parameters:
        - L: The length of the delay in calls to forward().
        - D: The dimensionality of the data in the input and output tensors.
        """
        super(DelayLine, self).__init__()
        self.L = L
        self.D = D
        self.buffer = torch.zeros((L, D))  # Initialize the circular buffer
        self.index = 0  # Current index for inserting the next tensor
        self.call_count = 0  # Count how many times forward() has been called

    def forward(self, x):
        """
        Forward pass of the DelayLine.

        Parameters:
        - x: The input tensor of shape (B, D).

        Returns:
        - A newly created output tensor of shape (B, D). It will be zeros for the first L calls,
          after which it will be a copy of the input tensor from L calls ago.
        """
        # Check if the input tensor has the expected shape
        # L, D = x.size() # batch size, sequence length, embedding dimensionality
        # assert L == (self.L), f"Input tensor does not match expected length {L=}."
        # assert D == (self.D), f"Input tensor does not match expected dimensionality {D=}."
        print(f"DelayLine::forward: {x.shape=}")

        # Determine the output
        if self.call_count < self.L:
            # For the first L-1 calls, output zeros
            output = torch.zeros_like(x)
        else:
            # Starting from the Lth call, output the tensor from L calls ago
            # The output should be retrieved before updating the buffer
            output = self.buffer[self.index].clone()  # Use .clone() to ensure a true copy is made

        # Insert the current input tensor into the buffer at the current index
        self.buffer[self.index] = x.detach() # detach turns off gradient propagation here

        # Update the index, wrapping around to create a circular buffer
        self.index = (self.index + 1) % self.L

        # Increment the call count
        self.call_count += 1

        return output

"""
# Example usage
L = 3  # Delay length
D = 4  # Dimensionality of the data
B = 2  # Batch size

print(f"\n{L=}, {D=}, {B=}\n")

delay_line = DelayLine(L, D)

# Simulate multiple calls to forward()
for i in range(5):
    x = torch.randn(B, D)  # Random input tensor
    print(f"Call {i+1} Input:\n{x}\n")
    output = delay_line(x)
    print(f"Call {i+1} Output:\n{output}\n\n")
"""
