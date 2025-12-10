"""
Analysis of weight and bias matrix shapes for a feedforward neural network.

Network Architecture:
- Input Layer: 3 features (n_x = 3)
- Layer 1: 4 units (n_1 = 4)
- Layer 2: 3 units (n_2 = 3)
- Layer 3: 1 unit (n_3 = 1) - Output layer
"""

import numpy as np

print("=" * 70)
print("NEURAL NETWORK ARCHITECTURE")
print("=" * 70)
print("Input Layer: 3 features (n_x = 3)")
print("Layer 1: 4 units (n_1 = 4)")
print("Layer 2: 3 units (n_2 = 3)")
print("Layer 3: 1 unit (n_3 = 1) - Output")
print()

print("=" * 70)
print("RULE: For layer l connecting layer (l-1) to layer l")
print("=" * 70)
print("W^[l] shape = (n_l, n_{l-1})")
print("b^[l] shape = (n_l, 1)")
print()

print("=" * 70)
print("CALCULATING SHAPES")
print("=" * 70)

# Layer 1: connects input (n_x=3) to layer 1 (n_1=4)
print("\nLayer 1:")
print(f"  W^[1] connects input (n_x=3) → Layer 1 (n_1=4)")
print(f"  W^[1] shape = (n_1, n_x) = (4, 3) ✓")
print(f"  b^[1] shape = (n_1, 1) = (4, 1) ✓")

# Layer 2: connects layer 1 (n_1=4) to layer 2 (n_2=3)
print("\nLayer 2:")
print(f"  W^[2] connects Layer 1 (n_1=4) → Layer 2 (n_2=3)")
print(f"  W^[2] shape = (n_2, n_1) = (3, 4) ✓")
print(f"  b^[2] shape = (n_2, 1) = (3, 1) ✓")

# Layer 3: connects layer 2 (n_2=3) to output (n_3=1)
print("\nLayer 3:")
print(f"  W^[3] connects Layer 2 (n_2=3) → Output (n_3=1)")
print(f"  W^[3] shape = (n_3, n_2) = (1, 3) ✓")
print(f"  b^[3] shape = (n_3, 1) = (1, 1) ✓")

print("\n" + "=" * 70)
print("EVALUATING STATEMENTS")
print("=" * 70)

statements = [
    ("W^[1] will have shape (4, 3)", True, "✓ CORRECT"),
    ("W^[1] will have shape (3, 4)", False, "✗ WRONG - This is transposed"),
    ("b^[1] will have shape (3, 1)", False, "✗ WRONG - Should be (4, 1)"),
    ("b^[1] will have shape (4, 1)", True, "✓ CORRECT"),
    ("b^[1] will have shape (1, 4)", False, "✗ WRONG - Should be (4, 1)"),
    ("W^[2] will have shape (4, 3)", False, "✗ WRONG - Should be (3, 4)"),
    ("W^[2] will have shape (3, 1)", False, "✗ WRONG - Should be (3, 4)"),
    ("W^[2] will have shape (3, 4)", True, "✓ CORRECT"),
    ("W^[2] will have shape (1, 3)", False, "✗ WRONG - That's W^[3]"),
]

print("\nTrue Statements (check all that apply):")
print("-" * 70)
for statement, is_correct, explanation in statements:
    status = "✓ TRUE" if is_correct else "✗ FALSE"
    print(f"{status:10} | {statement:35} | {explanation}")

print("\n" + "=" * 70)
print("WHY THESE SHAPES?")
print("=" * 70)
print("""
The forward propagation formula for layer l is:
    z^[l] = W^[l] * a^[l-1] + b^[l]

For this to work:
- a^[l-1] has shape (n_{l-1}, m) where m = number of examples
- W^[l] must have shape (n_l, n_{l-1}) to multiply correctly
- b^[l] must have shape (n_l, 1) to broadcast correctly
- Result z^[l] has shape (n_l, m)

Example for Layer 1:
- Input a^[0] (which is X): shape (3, m)
- W^[1]: shape (4, 3)
- W^[1] * a^[0]: (4, 3) × (3, m) = (4, m) ✓
- b^[1]: shape (4, 1) broadcasts to (4, m) ✓
""")

# Demonstrate with actual matrix multiplication
print("=" * 70)
print("VERIFICATION WITH ACTUAL MATRICES")
print("=" * 70)

# Simulate with m=2 examples
m = 2
X = np.random.randn(3, m)  # Input: (3, 2)

# Initialize parameters
W1 = np.random.randn(4, 3) * 0.01
b1 = np.random.randn(4, 1) * 0.01
W2 = np.random.randn(3, 4) * 0.01
b2 = np.random.randn(3, 1) * 0.01
W3 = np.random.randn(1, 3) * 0.01
b3 = np.random.randn(1, 1) * 0.01

print(f"\nInput X shape: {X.shape}")
print(f"W^[1] shape: {W1.shape}, b^[1] shape: {b1.shape}")
print(f"W^[2] shape: {W2.shape}, b^[2] shape: {b2.shape}")
print(f"W^[3] shape: {W3.shape}, b^[3] shape: {b3.shape}")

# Forward propagation
Z1 = W1 @ X + b1  # (4, 3) @ (3, 2) + (4, 1) = (4, 2)
A1 = np.maximum(0, Z1)  # ReLU activation
print(f"\nAfter Layer 1: Z1 shape = {Z1.shape}, A1 shape = {A1.shape}")

Z2 = W2 @ A1 + b2  # (3, 4) @ (4, 2) + (3, 1) = (3, 2)
A2 = np.maximum(0, Z2)  # ReLU activation
print(f"After Layer 2: Z2 shape = {Z2.shape}, A2 shape = {A2.shape}")

Z3 = W3 @ A2 + b3  # (1, 3) @ (3, 2) + (1, 1) = (1, 2)
A3 = 1 / (1 + np.exp(-Z3))  # Sigmoid activation
print(f"After Layer 3: Z3 shape = {Z3.shape}, A3 shape = {A3.shape}")

print("\n✓ All matrix multiplications work correctly with these shapes!")







