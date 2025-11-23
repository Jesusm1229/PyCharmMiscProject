"""Test script to verify all imports work correctly"""

print("Testing imports...")
print("=" * 50)

try:
    import numpy as np
    print("OK: numpy imported")
except Exception as e:
    print(f"FAILED: numpy failed: {e}")

try:
    import copy
    print("OK: copy imported")
except Exception as e:
    print(f"FAILED: copy failed: {e}")

try:
    import sklearn
    import sklearn.datasets
    import sklearn.linear_model
    print("OK: sklearn imported")
except Exception as e:
    print(f"FAILED: sklearn failed: {e}")

try:
    from planar_utils import load_planar_dataset, sigmoid
    print("OK: planar_utils imported")
except Exception as e:
    print(f"FAILED: planar_utils failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("OK: matplotlib imported")
except Exception as e:
    print(f"FAILED: matplotlib failed: {e}")

try:
    from testCases_v2 import *
    print("OK: testCases_v2 imported")
except Exception as e:
    print(f"FAILED: testCases_v2 failed: {e}")

try:
    from public_tests import *
    print("OK: public_tests imported")
except Exception as e:
    print(f"FAILED: public_tests failed: {e}")

print("=" * 50)
print("All imports completed!")

