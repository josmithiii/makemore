import torch
import torch.nn as nn

class DelayList(nn.Module):
    def __init__(self, L):
        super(DelayList, self).__init__()
        self.L = L
        self.buffer = [] # list of input buffers
        self.index = 0

    def forward(self, x):
        # self.buffer.append(x.clone().detach()) # detached copy of x added to our delay-line list
        self.buffer.append(x) # append x reference to our delay-line list
        if len(self.buffer) > self.L:
            # If the buffer size exceeds the delay length, pop the earliest tensor
            output = self.buffer.pop(0)
        else:
            # Return zeros for the first L-1 calls
            output = torch.zeros_like(x)

        return output

"""
# Example usage
L = 3  # Delay length
D = 4  # Dimensionality of the data
B = 2  # Batch size

print(f"\n{L=}, {D=}, {B=}\n")

delay_line = DelayList(L)

# Simulate multiple calls to forward()
for i in range(5):
    x = torch.randn(B, D)  # Random input tensor
    print(f"Call {i+1} Input:\n{x}\n")
    output = delay_line(x)
    print(f"Call {i+1} Output:\n{output}\n\n")
"""

# ------------------------------------------------------------------------------
import gc
# prints currently alive Tensors and Variables
def printObjects():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass
