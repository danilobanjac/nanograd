# Nanograd: A Gentle Introduction to Automatic Differentiation

## 1. Overview

**Nanograd** is a tiny Python library that demonstrates how automatic differentiation (autodiff) can be implemented from scratch. Autodiff is a technique that allows a computer program to automatically compute **derivatives** (i.e., “slopes”) of functions that are defined in code. In the context of **machine learning**, these derivatives are crucial to performing **gradient-based optimization** (e.g., training neural networks).

### Key points we’ll explore

1. **What is a derivative?** (Refresher from high school math)  
2. **Why do we need derivatives in machine learning?**  
3. **How does Nanograd represent a function as a “graph”?**  
4. **How does the chain rule enable “automatic” backpropagation?**  
5. **Step-by-step examples** of simple functions and how Nanograd calculates their derivatives.  
6. **Applying these ideas to logistic regression** (the AND-gate example).

---

## 2. Quick Refresher: What Is a Derivative?

In simple terms, the **derivative** of a function $`f(x)`$ with respect to $`x`$ (written $`\frac{d f}{dx}`$) is the **rate of change** or **slope** of $`f`$ when we slightly change $`x`$.

- For example, if $`f(x) = x^2`$, the derivative is $`\frac{d}{dx}(x^2) = 2x`$.
- If $`x`$ changes by a tiny amount, the function $`x^2`$ will change by approximately $`2x \times (\text{tiny amount})`$.

Why do we care? In **machine learning**, we often want to **adjust parameters** (like weights in a neural network) to minimize some “loss” function. We do this by following the slope (or **gradient**) that tells us **which direction** reduces the loss.

---

## 3. Extending to More Dimensions: Partial Derivatives

When we have multiple variables, say $`x, y, z`$, a function might look like

$$
f(x, y, z) = x \times y + z.
$$

We measure how $`f`$ changes in each variable’s direction separately:

- $`\frac{\partial f}{\partial x}`$ answers: “How does $`f`$ change if I only move $`x`$ a little, but keep $`y`$ and $`z`$ fixed?”
- $`\frac{\partial f}{\partial y}`$, $`\frac{\partial f}{\partial z}`$ are interpreted similarly.

In practice, a neural network has many parameters (thousands or millions), and we need all their partial derivatives to do a **gradient descent** update.

---

## 4. The Chain Rule: Linking Functions Together

The **chain rule** is what allows us to compute the derivative of complicated “chained” functions without re-deriving everything from scratch. A classic single-variable example:

$$
\text{If } f(x) = (3x + 2)^2, \quad 
\text{we can set } u(x) = 3x + 2, \quad
\text{so } f(x) = u(x)^2.
$$

Using the chain rule:

$$
\frac{df}{dx} = \frac{df}{du} \times \frac{du}{dx}.
$$

- We know $`\frac{df}{du} = 2u`$, and $`\frac{du}{dx} = 3`$.  
- Substituting $`u = 3x + 2`$, we get $`\frac{df}{dx} = 2(3x + 2) \times 3 = 6(3x + 2)`$.

If you expand $`(3x + 2)^2`$ yourself (to $`9x^2 + 12x + 4`$), you can also confirm $`\frac{d}{dx}(9x^2 + 12x + 4) = 18x + 12`$. Factoring out 6, that’s $`6(3x + 2)`$. The chain rule just organizes these steps.

When we build large functions (like neural networks), we apply the chain rule **many times** to handle each “layer” or operation.

---

## 5. Nanograd’s Approach: Representing a Function as a Graph

### 5.1. The “Value” Nodes

In Nanograd, every number we use in our computation is wrapped in a `Value` object. For instance:

```python
from nanograd.nanograd import Value

x = Value(2.0)
y = Value(3.0)
z = x * y  # Another Value representing 2.0 * 3.0
```

- Each `Value` knows:
  1. Its **data** (the actual numeric value).
  2. Its **parents**: other `Value`s that were combined to create it.
  3. A **grad** attribute that will store $`\frac{\partial \text{(output)}}{\partial \text{(this Value)}}`$.
  4. A function (`grad_fn`) that knows how to compute its partial derivatives when we backprop.

### 5.2. Building the “Computation Graph”

When you do `z = x * y`, Nanograd internally creates a small “node” in a graph where:

- $`z`$ depends on $`x`$ and $`y`$.  
- The operation is `*`.  
- If we change `x` or `y`, `z` will change according to $`\frac{\partial z}{\partial x} = y`$ and $`\frac{\partial z}{\partial y} = x`$.

Nanograd keeps track of these relationships automatically.

---

## 6. The Backward Pass: Applying the Chain Rule in Reverse

After you define your final output function (say a “loss” in machine learning), you call `some_value.backward()`. Nanograd then:

1. Sets the gradient of `some_value` (the final output) to 1.0 (because $`\frac{\partial (\text{itself})}{\partial (\text{itself})} = 1`$).  
2. Looks at each node’s parents in the correct topological order (ensuring we visit a parent node *before* its children).  
3. Applies the appropriate chain rule step to update each parent’s gradient.  

This is often called **reverse-mode autodiff**, a.k.a. “backpropagation,” and it is what we do in neural networks.

---

## 7. Putting It All Together with a Simple Example

Let’s illustrate with a small expression:

$$
z = (x \times y + 3)^2.
$$

1. **Forward pass** (compute the actual numeric value). Suppose $`x = 2, y = 3`$.

   $$x \times y = 2 \times 3 = 6, \quad 6 + 3 = 9, \quad 9^2 = 81.$$

   So $`z = 81`$.

2. **Backward pass** (compute partial derivatives $`\frac{\partial z}{\partial x}`$ and $`\frac{\partial z}{\partial y}`$ automatically):

   - We can do it by hand, or rely on Nanograd’s chain rule. By hand:

     $$
     z = (xy + 3)^2, \quad
     \frac{\partial z}{\partial (xy + 3)} = 2(xy + 3), \quad
     \frac{\partial (xy + 3)}{\partial x} = y, \quad
     \frac{\partial (xy + 3)}{\partial y} = x.
     $$

     So, 
     $$
     \frac{\partial z}{\partial x} 
     = 2(xy + 3) \times y = 2(6 + 3) \times 3 = 2 \times 9 \times 3 = 54.
     $$
     $$
     \frac{\partial z}{\partial y} 
     = 2(xy + 3) \times x = 2(9) \times 2 = 36.
     $$

   - In Nanograd, you’d just do:
     ```python
     x = Value(2.0)
     y = Value(3.0)
     z = ((x * y) + Value(3.0)) ** 2  # squared
     z.backward()

     print("dz/dx:", x.grad)  # should be 54
     print("dz/dy:", y.grad)  # should be 36
     ```
     Nanograd automatically obtains the same result.

---

## 8. A Peek at the Code: `grad_fn` and Topological Sort

In **nanograd**:

- Each operation (like `+`, `*`, etc.) is **decorated** by a rule telling us how to compute its local derivative. For multiplication $`x \times y`$:
  $$
  \frac{\partial}{\partial x}(x \times y) = y, \quad
  \frac{\partial}{\partial y}(x \times y) = x.
  $$
- When you create a new `Value` (call it `out`) from two parents (`lhs` and `rhs`), `out`’s `grad_fn` is set so that it knows:  
  1. **Which** partial derivatives to add to `lhs.grad` and `rhs.grad` once `out.grad` is known.  
  2. The exact chain rule formula for that operation.

When `backward()` is called on a final node:
1. Nanograd sets `final_node.grad = 1.0`.
2. It does a **topological sort** to find the correct order to process nodes in reverse.  
3. Each node runs its `grad_fn`, updating the gradient of its parents.  
4. This propagates all the way to the leaves (parameters or input variables).

---

## 9. Example: Logistic Regression for AND

We can see how this works in a **binary classification** scenario. Let’s revisit the code snippet that learns the **AND function**:

```python
import random
from nanograd.nanograd import Value

# Data: AND truth table
training_data = [
    ((0, 0), 0),
    ((0, 1), 0),
    ((1, 0), 0),
    ((1, 1), 1),
]

# Random initialization of weights/bias
w1 = Value(random.uniform(-1, 1), "w1")
w2 = Value(random.uniform(-1, 1), "w2")
b  = Value(random.uniform(-1, 1), "b")

def forward(x1: Value, x2: Value) -> Value:
    # logistic regression: sigmoid(w1*x1 + w2*x2 + b)
    return (x1 * w1 + x2 * w2 + b).sigmoid()

def cross_entropy(y_true: Value, y_pred: Value) -> Value:
    # -[ y_true * ln(y_pred) + (1-y_true)*ln(1-y_pred) ]
    return -(y_true * y_pred.ln() + (1-y_true) * (1-y_pred).ln())

# Training
epochs = 10_000
lr = 0.05

for epoch in range(epochs):
    total_loss = Value(0.0)
    
    # Accumulate loss over 4 examples
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
    
    # Gradient descent update
    w1.data -= lr * w1.grad
    w2.data -= lr * w2.grad
    b.data  -= lr * b.grad

# Evaluate final predictions
for (x1_raw, x2_raw), _ in training_data:
    y_pred = forward(Value(x1_raw), Value(x2_raw))
    print(f"AND({x1_raw}, {x2_raw}) ≈ {y_pred.data:.4f}")
```

### 9.1. Why Are We Taking Derivatives Here?

- **Logistic regression** tries to find parameters $`w_1, w_2, b`$ that minimize the overall “loss” (the cross-entropy), i.e., how “wrong” our predictions are.  
- Each iteration, we compute **partial derivatives** of the loss with respect to each parameter ($`\frac{\partial L}{\partial w_1}`$, etc.).  
- We adjust $`w_1`$ by subtracting a small fraction of $`\frac{\partial L}{\partial w_1}`$. This is exactly **gradient descent**.

### 9.2. Sigmoid and Cross-Entropy

- $`\text{sigmoid}(z) = \frac{1}{1 + e^{-z}}`$  
  - The derivative is $`\sigma'(z) = \sigma(z) [1 - \sigma(z)]`$.  
- $`\ln`$ is the **natural logarithm**. Its derivative is $`1 / x`$.  
- So each time we do a logistic regression forward pass plus cross-entropy, we rely on multiple small chain-rule steps that Nanograd automatically wires up.

---

## 10. Summary

1. **Derivatives** measure how a function changes with respect to its inputs.  
2. **Partial derivatives** extend this idea to multiple variables.  
3. **Chain rule** is the secret sauce that lets us break down complex operations into simpler steps.  
4. **Nanograd** uses a directed graph of `Value` nodes to track how each output depends on its parents.  
5. **Reverse-mode autodiff** (“backprop”) systematically accumulates gradients from the final output back to the inputs/parameters.  
6. **Machine learning**: We use these gradients to perform **gradient descent** updates on model parameters, eventually finding parameter values that (hopefully) minimize our loss function.

---

## 11. Additional Examples to Build Confidence

### 11.1. Simple Single Variable

```python
x = Value(5.0, name="x")
y = (x + 3) * (x + 2)
y.backward()

# Expected derivative at x=5 for y = (x+3)*(x+2)
# Expand: y = (x+3)(x+2) = x^2 + 5x + 6
# dy/dx = 2x + 5
# at x=5 -> dy/dx = 2*5 + 5 = 15
print("dy/dx:", x.grad)  # Should be 15
```

### 11.2. Two Variables

```python
a = Value(2.0, name="a")
b = Value(-4.0, name="b")
c = (a - b) ** 2
c.backward()

# c = (a - b)^2
# partial derivative wrt a: 2(a - b)
# partial derivative wrt b: -2(a - b)

print("dc/da:", a.grad)  # Should be 2(a-b) = 2(2 - (-4)) = 2*6 = 12
print("dc/db:", b.grad)  # Should be -2(a-b) = -2(6) = -12
```

---

## Closing Thoughts

- **Nanograd** is perfect for learning how gradients can be computed automatically in code. Real frameworks like **PyTorch**, **TensorFlow**, and **JAX** implement the same concepts but are far more optimized and have many additional features.
- Understanding these basics is enough to see how we can build **larger neural networks**, do **backprop**, and train models effectively.

**Congratulations** on making it through this detailed explanation! If you grasp these fundamentals:
- You know how derivatives work in code.  
- You understand the chain rule and backprop.  
- You can see how gradient updates optimize neural network weights.

You’re well on your way to exploring deeper machine learning topics.

---

### Additional Resources

- **3Blue1Brown:** Has excellent animated videos on calculus, neural networks, and backpropagation.  
- **PyTorch or TensorFlow** tutorials: If you want to see how large frameworks do the same thing under the hood.  
- **Further reading** on the chain rule, partial derivatives, and linear algebra is helpful as you progress to more complex neural nets.
