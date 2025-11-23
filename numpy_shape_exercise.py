"""
Numpy Shape Exercise - Understanding np.sum with axis and keepdims

Question:
A = np.random.randn(4,3)
B = np.sum(A, axis = 1, keepdims = True)
What will be B.shape?
"""

import numpy as np

# Create the array
A = np.random.randn(4, 3)

print("=" * 60)
print("NUMPY SHAPE EXERCISE")
print("=" * 60)

print(f"\nA = np.random.randn(4, 3)")
print(f"A.shape = {A.shape}")
print(f"\nA = \n{A}")

# Perform the sum operation
B = np.sum(A, axis=1, keepdims=True)

print("\n" + "-" * 60)
print("B = np.sum(A, axis=1, keepdims=True)")
print("-" * 60)

print(f"\nB.shape = {B.shape}")
print(f"\nB = \n{B}")

print("\n" + "=" * 60)
print("EXPLANATION:")
print("=" * 60)
print("""
1. A has shape (4, 3) - 4 rows, 3 columns

2. np.sum(A, axis=1, keepdims=True):
   - axis=1 means we sum along the columns (the second dimension)
   - For each row, we sum the 3 column values
   - This gives us 4 sums (one per row)
   - keepdims=True preserves the 2D shape

3. Result:
   - Without keepdims: shape would be (4,) - a 1D array
   - With keepdims=True: shape is (4, 1) - a 2D array with 4 rows and 1 column
""")

print("\n" + "=" * 60)
print("ANSWER: B.shape = (4, 1)")
print("=" * 60)

# Additional demonstration: compare with and without keepdims
print("\n" + "-" * 60)
print("COMPARISON: With vs Without keepdims")
print("-" * 60)

B_without = np.sum(A, axis=1, keepdims=False)
print(f"\nB_without = np.sum(A, axis=1, keepdims=False)")
print(f"B_without.shape = {B_without.shape}")
print(f"B_without = {B_without}")

print(f"\nB_with_keepdims.shape = {B.shape}")
print(f"B_with_keepdims = \n{B}")

print("\n" + "=" * 60)
print("KEY TAKEAWAY:")
print("=" * 60)
print("""
- axis=0: Sum along rows (first dimension) → reduces rows
- axis=1: Sum along columns (second dimension) → reduces columns
- keepdims=True: Maintains the number of dimensions (2D stays 2D)
- keepdims=False: Reduces dimensions (2D becomes 1D)
""")

