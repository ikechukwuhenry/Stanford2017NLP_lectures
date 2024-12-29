import torch
import torch.nn as nn

# Import pprint, module we use for making our print statment prettier
import pprint
pp = pprint.PrettyPrinter()


list_of_lists = [
    [1, 2, 3],
    [4, 5, 6],
]

data = torch.tensor(list_of_lists)
print(data)

# Initializing a tensor with an explicit data type
data = torch.tensor([
    [0, 1],
    [2, 3],
    [4, 5],
    ], dtype=torch.float32)
print(data)

print("shape of the data ", data.shape)

# Reshape tensors
rr = torch.arange(1, 16)
print(rr)
print("shape of rr is ", rr.shape)

# Reshape the tenosr to 5 X 3
rr = rr.view(5, 3)
print(rr)
print("new shape after reshaping is ", rr.shape)

assignment = torch.tensor([
                [1, 2.2, 9.6],
                [4, -7.2, 6.3]
            ])
# get first column
# print(assignment[-1])
# Get first row
print(assignment[0])

# Calculate gradient using back prop
# Requires_grad parameter tells Pytorch to store gradients
x = torch.tensor([2.], requires_grad=True)

# Print the gradient if it is calculated
# Currently None since x is a scalar
pp.pprint(x.grad)

# Calculate gradient of y with respect to x
y = x * x * 3  # 3x^2
y.backward()
pp.pprint(x.grad) # d(y)/d(x) = d(3x^2)/d(x) = 6x = 12

# Let's run backprop from a different tensor
z = x * x * 3 # 3x^2
z.backward()
pp.pprint(x.grad)

# Reset grad
x.grad = None
z = x * x * 3 # 3x^2
z.backward()
# y = 3x^2
pp.pprint(x.grad)

z = x * x * 3 # 3x^2
z.backward()
pp.pprint(x.grad)
z = x * x * 3 # 3x^2
z.backward()
pp.pprint(x.grad)
