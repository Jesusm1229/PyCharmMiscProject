"""
Analysis of parameter initialization options for neural network
Given: layer_dims = [nx, 4, 3, 2, 1]
"""

import numpy as np

# Expected shapes:
# W[1]: (4, nx)   b[1]: (4, 1)
# W[2]: (3, 4)    b[2]: (3, 1)
# W[3]: (2, 3)    b[3]: (2, 1)
# W[4]: (1, 2)    b[4]: (1, 1)

layer_dims = [10, 4, 3, 2, 1]  # Using nx=10 as example
parameter = {}

print("=" * 70)
print("OPTION 1: for i in range(1, len(layer_dims)/2):")
print("=" * 70)
print(f"len(layer_dims) = {len(layer_dims)}")
print(f"len(layer_dims)/2 = {len(layer_dims)/2}")
print("❌ PROBLEM: range() requires integers, but len(layer_dims)/2 = 2.5 (float)")
print("❌ Even if it worked, would only iterate i=1, missing layers 2, 3, 4")
print()

print("=" * 70)
print("OPTION 2: for i in range(len(layer_dims)-1):")
print("=" * 70)
parameter2 = {}
for i in range(len(layer_dims)-1):  # i = 0, 1, 2, 3
    parameter2['W' + str(i+1)] = np.random.randn(layer_dims[i], layer_dims[i+1]) * 0.01
    parameter2['b' + str(i+1)] = np.random.randn(layer_dims[i+1], 1) * 0.01
    print(f"i={i}: W[{i+1}] shape = ({layer_dims[i]}, {layer_dims[i+1]}) = {parameter2['W' + str(i+1)].shape}")
    print(f"      b[{i+1}] shape = ({layer_dims[i+1]}, 1) = {parameter2['b' + str(i+1)].shape}")
print()
print("Expected W[1] shape: (4, 10) but got:", parameter2['W1'].shape)
print("❌ WRONG: W[1] should be (4, nx) but got (nx, 4)")
print()

print("=" * 70)
print("OPTION 3: for i in range(len(layer_dims)):")
print("=" * 70)
parameter3 = {}
try:
    for i in range(len(layer_dims)):  # i = 0, 1, 2, 3, 4
        parameter3['W' + str(i+1)] = np.random.randn(layer_dims[i+1], layer_dims[i]) * 0.01
        parameter3['b' + str(i+1)] = np.random.randn(layer_dims[i+1], 1) * 0.01
        print(f"i={i}: W[{i+1}] shape = ({layer_dims[i+1]}, {layer_dims[i]}) = {parameter3['W' + str(i+1)].shape}")
        print(f"      b[{i+1}] shape = ({layer_dims[i+1]}, 1) = {parameter3['b' + str(i+1)].shape}")
except IndexError as e:
    print(f"❌ ERROR: {e}")
    print("When i=4, layer_dims[i+1] = layer_dims[5] doesn't exist!")
print()

print("=" * 70)
print("OPTION 4: for i in range(len(layer_dims)-1):")
print("=" * 70)
parameter4 = {}
for i in range(len(layer_dims)-1):  # i = 0, 1, 2, 3
    parameter4['W' + str(i+1)] = np.random.randn(layer_dims[i+1], layer_dims[i]) * 0.01
    parameter4['b' + str(i+1)] = np.random.randn(layer_dims[i+1], 1) * 0.01
    print(f"i={i}: W[{i+1}] shape = ({layer_dims[i+1]}, {layer_dims[i]}) = {parameter4['W' + str(i+1)].shape}")
    print(f"      b[{i+1}] shape = ({layer_dims[i+1]}, 1) = {parameter4['b' + str(i+1)].shape}")
print()
print("Expected shapes:")
print("  W[1]: (4, 10) ✓")
print("  W[2]: (3, 4)  ✓")
print("  W[3]: (2, 3)  ✓")
print("  W[4]: (1, 2)  ✓")
print("✅ CORRECT!")
print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("Option 1: ❌ Syntax error (float in range)")
print("Option 2: ❌ Wrong weight dimensions (transposed)")
print("Option 3: ❌ Index out of bounds error")
print("Option 4: ✅ CORRECT - Properly initializes all parameters")







