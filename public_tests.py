"""
Public test functions for the neural network assignment.
These are simplified versions that check basic functionality.
"""

import numpy as np

def layer_sizes_test(function):
    """Test layer_sizes function"""
    try:
        from testCases_v2 import layer_sizes_test_case
        X, Y = layer_sizes_test_case()
        n_x, n_h, n_y = function(X, Y)
        assert n_x == 5, f"Expected n_x=5, got {n_x}"
        assert n_h == 4, f"Expected n_h=4, got {n_h}"
        assert n_y == 2, f"Expected n_y=2, got {n_y}"
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")

def initialize_parameters_test(function):
    """Test initialize_parameters function"""
    try:
        from testCases_v2 import initialize_parameters_test_case
        n_x, n_h, n_y = initialize_parameters_test_case()
        parameters = function(n_x, n_h, n_y)
        assert parameters['W1'].shape == (n_h, n_x), f"W1 shape incorrect"
        assert parameters['b1'].shape == (n_h, 1), f"b1 shape incorrect"
        assert parameters['W2'].shape == (n_y, n_h), f"W2 shape incorrect"
        assert parameters['b2'].shape == (n_y, 1), f"b2 shape incorrect"
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")

def forward_propagation_test(function):
    """Test forward_propagation function"""
    try:
        from testCases_v2 import forward_propagation_test_case
        X, parameters = forward_propagation_test_case()
        A2, cache = function(X, parameters)
        assert A2.shape == (1, X.shape[1]), f"A2 shape incorrect"
        assert 'Z1' in cache and 'A1' in cache and 'Z2' in cache and 'A2' in cache
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")

def compute_cost_test(function):
    """Test compute_cost function"""
    try:
        from testCases_v2 import compute_cost_test_case
        A2, Y = compute_cost_test_case()
        cost = function(A2, Y)
        assert isinstance(cost, (float, np.floating)), "Cost should be a float"
        assert cost > 0, "Cost should be positive"
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")

def backward_propagation_test(function):
    """Test backward_propagation function"""
    try:
        from testCases_v2 import backward_propagation_test_case
        parameters, cache, X, Y = backward_propagation_test_case()
        grads = function(parameters, cache, X, Y)
        assert 'dW1' in grads and 'db1' in grads and 'dW2' in grads and 'db2' in grads
        assert grads['dW1'].shape == parameters['W1'].shape
        assert grads['dW2'].shape == parameters['W2'].shape
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")

def update_parameters_test(function):
    """Test update_parameters function"""
    try:
        from testCases_v2 import update_parameters_test_case
        parameters, grads = update_parameters_test_case()
        updated = function(parameters, grads)
        assert 'W1' in updated and 'b1' in updated and 'W2' in updated and 'b2' in updated
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")

def nn_model_test(function):
    """Test nn_model function"""
    try:
        np.random.seed(1)
        X = np.random.randn(2, 400)
        Y = np.random.randint(0, 2, (1, 400))
        parameters = function(X, Y, n_h=5, num_iterations=1000, print_cost=False)
        assert 'W1' in parameters and 'b1' in parameters and 'W2' in parameters and 'b2' in parameters
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")

def predict_test(function):
    """Test predict function"""
    try:
        from testCases_v2 import predict_test_case
        parameters, X = predict_test_case()
        predictions = function(parameters, X)
        assert predictions.shape == (1, X.shape[1]) or predictions.shape == (X.shape[1],)
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")












