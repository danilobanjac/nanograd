# Nanograd

A minimal, educational automatic differentiation (autodiff) library in Python, with a toy neural network example and a logistic regression “AND” classifier.

## Table of Contents

- [Overview](#overview)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Quick Start](#quick-start)  
  - [1. The AND Example (Logistic Regression)](#1-the-and-example-logistic-regression)  
  - [2. Visualizing the Computation Graph](#2-visualizing-the-computation-graph)  
- [How It Works](#how-it-works)  
- [License](#license)

---

## Overview

**Nanograd** is a minimalistic autodiff library demonstrating how to:
1. Represent arithmetic operations as nodes in a computation graph,
2. Compute gradients via reverse-mode automatic differentiation (backpropagation),
3. Build small neural networks (e.g., a Perceptron and Multi-Layer Perceptron) for illustrative purposes.

The library is composed of:
- A `Value` class that wraps Python floats and tracks gradients,
- Operator overloading for common arithmetic and activation functions,
- Basic building blocks for feed-forward networks.

---

## Requirements

- **Python >= 3.12**  
  Some of the examples use newer Python features (e.g., `itertools.pairwise`).  
- **Graphviz** (optional but recommended for visualization)  
  - On many Linux distributions, you can install it via `sudo apt-get install graphviz`.  
  - On Windows, download from the [official Graphviz website](https://graphviz.org/download/).  
  - For Mac, you can use Homebrew: `brew install graphviz`.  

Additionally, you’ll need the Python `graphviz` package to visualize your computation graph:

```
pip install graphviz
```

---

## Installation

1. **Clone or download** this repository:

   ```bash
   git clone https://github.com/<user>/nanograd.git
   cd nanograd
   ```

2. **(Optional) Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate     # On Linux/Mac
   venv\Scripts\activate        # On Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install graphviz
   # If you plan on using the MLP classes, no other dependencies are required
   # besides the standard library and 'random', 'itertools', etc.
   ```

---

## Quick Start

### 1. The AND Example (Logistic Regression)

Below is a simple script demonstrating how to learn the AND function using a single logistic neuron in **nanograd**. Create a file (e.g., `example_and.py`) with the following content:

```python
import random
from nanograd.nanograd import Value

# Training data for AND gate
training_data = [
    ((0, 0), 0),
    ((0, 1), 0),
    ((1, 0), 0),
    ((1, 1), 1),
]

# Initialize parameters
w1 = Value(random.uniform(-1, 1), "w1")
w2 = Value(random.uniform(-1, 1), "w2")
b  = Value(random.uniform(-1, 1), "b")

# Training hyperparameters
lr = 0.05
epochs = 10_000

def forward(x1: Value, x2: Value) -> Value:
    # Single-layer logistic regression: sigmoid(w1*x1 + w2*x2 + b)
    return (x1 * w1 + x2 * w2 + b).sigmoid()

def cross_entropy(y_true: Value, y_pred: Value) -> Value:
    # Binary cross-entropy: -(y_true * ln(y_pred) + (1-y_true)*ln(1-y_pred))
    return -(y_true * y_pred.ln() + (1 - y_true) * (1 - y_pred).ln())

for epoch in range(1, epochs + 1):
    # Accumulate total loss over the mini-dataset
    total_loss = Value(0.0)

    for (x1_raw, x2_raw), y_raw in training_data:
        x1 = Value(x1_raw)
        x2 = Value(x2_raw)
        y_true = Value(y_raw)

        y_pred = forward(x1, x2)
        total_loss += cross_entropy(y_true, y_pred)

    # Average loss
    avg_loss = total_loss / len(training_data)

    # Backprop
    avg_loss.zero_grad()
    avg_loss.backward()

    # Gradient descent step
    w1.data -= lr * w1.grad
    w2.data -= lr * w2.grad
    b.data  -= lr * b.grad

# Print learned parameters
print(f"Trained parameters:\n  w1={w1.data}, w2={w2.data}, b={b.data}\n")

# Print final predictions
for (x1_raw, x2_raw), _ in training_data:
    x1 = Value(x1_raw)
    x2 = Value(x2_raw)
    y_pred = forward(x1, x2)
    print(f"For ({x1_raw},{x2_raw}): {y_pred.data}")
```

Then run:

```bash
python example_and.py
```

You should see final predictions close to `0` for `(0,0), (0,1), (1,0)` and close to `1` for `(1,1)`.

### 2. Visualizing the Computation Graph

If you want to visualize the computation graph of a particular expression, ensure `graphviz` is installed. For example:

```python
from nanograd.nanograd import Value

x = Value(2.0, name="x")
y = Value(3.0, name="y")
z = (x * y + x / y).tanh()

dot = z.visualize()
dot.render("graph_output", format="png", cleanup=True)
```

You’ll get a `graph_output.png` file showing the nodes (`x`, `y`, intermediate ops) and edges representing the computational flow.

---

## How It Works

1. **Computation Graph Construction**  
   Each arithmetic operation (`+`, `*`, `tanh()`, etc.) creates a new `Value` node referencing its parents.  
2. **Topological Sort & Backprop**  
   Calling `z.backward()` on the final output `z` does a reverse traversal of the graph—accumulating gradients in each node’s `grad` attribute.  
3. **Parameter Updates**  
   After backprop, you can do `w.data -= lr * w.grad` to perform a gradient descent step.

For more details, see the docstrings in `nanograd` or browse the code in this repository.
