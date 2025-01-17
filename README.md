# Nanograd

A tiny, educational autodifferentiation (AD) library for Python, plus a simple Multi-Layer Perceptron (MLP) implementation.  
**Nanograd** uses Python's built-in `graphlib.TopologicalSorter` to perform reverse-mode automatic differentiation on a DAG of operations.  

## Features

- **Value** class for tracking data, gradients, and references to parent `Value` objects.  
- **Unary & Binary autodiff** via decorators:
  - `+`, `-`, `*`, `tanh`, and more.
- **Graph visualization** through `graphviz`.
- **Mini neural network library** (Perceptron, Layer, MultiLayerPerceptron).

## Installation

1. Clone this repository.
2. Ensure you have Python version >=3.12.
3. (Optional) Create a virtual environment:  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
   Make sure to include `graphviz` and any other packages in your `requirements.txt`.

## Getting Started

### Simple Example with `Value`

```python
from nanograd.nanograd import Value

# Define two Values
x = Value(3.0, name="x")
y = Value(-2.0, name="y")

# Build an expression
z = x * y + x + 2.0  #  => z = 3.0 * (-2.0) + 3.0 + 2.0
print(f"z.data = {z.data}")  
# z.data = 3.0 * (-2.0) + 3.0 + 2.0 = -6 + 3 + 2 = -1

# Backprop
z.backward()

# Inspect gradients
print(f"x.grad = {x.grad}")
print(f"y.grad = {y.grad}")

# Explanation:
# z = x*y + x + 2
# dz/dx = y + 1
# dz/dy = x
# So x.grad = -2 + 1 = -1
#    y.grad = 3
```

You can visualize the computation graph with:

```python
dot = z.visualize()
dot.render("simple_example", view=True)
```

### Example: Optimizing a Simple Function

Suppose you want to find a minimum for the function \((x + 2)^2\). We'll do a small gradient descent by hand:

```python
x = Value(0.0, name="x")

for step in range(50):
    x.zero_grad()               # reset gradient
    y = (x + 2.0) * (x + 2.0)    # forward pass
    y.backward()                # backprop
    lr = 0.1
    x.data -= lr * x.grad       # gradient descent update

print(f"x after optimization: {x.data}")
# It should converge near x = -2
```

### Example: Training a MiniLayerPerceptron (MLP)

We have three main classes for building neural networks:

- `Perceptron(n_inputs)`
- `Layer(n_inputs, n_perceptrons)`
- `MultiLayerPerceptron(n_inputs, n_outs)`

**Goal:** Learn a simple XOR function with a 2-layer MLP.

```python
from nanograd.nanograd import Value
from nanograd.nn import MultiLayerPerceptron

# Create a small MLP:
#   2 inputs -> 4 hidden -> 1 output
mlp = MultiLayerPerceptron(n_inputs=2, n_outs=[4, 1])

# Our XOR dataset
data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

# Training loop
lr = 0.1
for epoch in range(1000):
    loss = Value(0.0)
    for inputs, target in data:
        # Forward pass
        out = mlp(inputs)[0]  # single output
        # We'll compute a simple mean squared error
        diff = out - target
        sample_loss = diff * diff
        loss = loss + sample_loss

    # Zero gradients
    for p in mlp.parameters():
        p.zero_grad()

    # Backprop
    loss.backward()

    # Gradient descent update
    for p in mlp.parameters():
        p.data -= lr * p.grad

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, loss = {loss.data:.4f}")

# Check predictions
for inputs, target in data:
    out = mlp(inputs)[0]
    print(f"Inputs={inputs}, MLP output={out.data:.4f}, target={target}")
```

After enough epochs, you should see that the output is close to the expected XOR results (0.0 or 1.0).  

---

## Development

- **Adding New Operations**:  
  If you want to add new operations (e.g., division, exponent, log), you can create a function with a custom backward rule. Decorate your method with `@autodiff_binary_op(backward_rule)` or `@autodiff_unary_op(backward_rule)` and implement the partial derivatives.

- **Experiments**:  
  You can easily extend the `Layer` or `MultiLayerPerceptron` classes with new activation functions (just define them as `@autodiff_unary_op(...)` on `Value`).
