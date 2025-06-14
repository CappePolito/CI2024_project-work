# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# -------------------------------
# Define custom mathematical functions
# -------------------------------

def add(x, y):
    """Addition function that handles numpy arrays."""
    return np.add(x, y)

def sub(x, y):
    """Subtraction function that handles numpy arrays."""
    return np.subtract(x, y)

def mul(x, y):
    """Multiplication function that handles numpy arrays."""
    return np.multiply(x, y)

def custom_sin(x):
    """Sine function that handles numpy arrays."""
    return np.sin(x)

def custom_cos(x):
    """Cosine function that handles numpy arrays."""
    return np.cos(x)

def custom_exp(x):
    """Exponential function that handles numpy arrays safely."""
    # Clip x to prevent overflow
    clipped_x = np.clip(x, -100, 100)
    return np.exp(clipped_x)

def safe_div(x, y):
    """Division function that prevents division by zero."""
    return x / (y + 1e-6)  # Add small epsilon to avoid division by zero

def square(x):
    """Square function."""
    return x ** 2

def safe_exp(x):
    """Exponential function that prevents overflow."""
    # Clip x to a reasonable maximum value
    clipped_x = np.clip(x, -100, 100)  # Adjust these bounds as needed
    return np.exp(clipped_x)

def safe_log(x):
    """Protected logarithm: returns log(abs(x)+epsilon) to avoid log(0)."""
    return np.log(np.abs(x) + 1e-6)

def safe_sqrt(x):
    """Protected square root: returns sqrt(abs(x)) so the argument is non-negative."""
    return np.sqrt(np.abs(x))

def custom_tanh(x):
    """Hyperbolic tangent function that handles numpy arrays."""
    return np.tanh(x)

def safe_pow(x, y):
    """Protected power function: returns abs(x) raised to the power y safely.
       This version clips the base and avoids division by zero issues."""
    epsilon = 1e-6
    # Take the absolute value and replace near-zero values with epsilon
    base = np.abs(x)
    base = np.where(base < epsilon, epsilon, base)
    # Clip the base to avoid extremely large numbers that cause overflow.
    base = np.clip(base, 0, 100)
    return np.power(base, y)

def cube(x):
    """Cube function."""
    return x ** 3

def reciprocal(x):
    """Reciprocal function with protection against division by zero."""
    return 1.0 / (x + 1e-6)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def gaussian(x):
    """Gaussian function."""
    return np.exp(-x ** 2)

def relu(x):
    """Rectified Linear Unit function."""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU function."""
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    """Exponential Linear Unit function."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x):
    """Swish activation function."""
    return x * sigmoid(x)

def mish(x):
    """Mish activation function."""
    return x * np.tanh(np.log1p(np.exp(x)))


def sin1_over_x(x):
    """Sine of 1/x function with protection against division by zero."""
    return np.sin(1.0 / (x + 1e-6))

def sinc(x):
    """Sinc function."""
    return np.sinc(x / np.pi)

def sawtooth(x):
    """Sawtooth wave function."""
    return 2 * (x / (2 * np.pi) - np.floor(0.5 + x / (2 * np.pi)))

def triangle_wave(x):
    """Triangle wave function."""
    return 2 * np.abs(2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5))) - 1

def square_wave(x):
    """Square wave function."""
    return np.sign(np.sin(x))


def bent_identity(x):
    """Bent identity function."""
    return (np.sqrt(x ** 2 + 1) - 1) / 2 + x

def softsign(x):
    """Softsign function."""
    return x / (1 + np.abs(x))

def hard_sigmoid(x):
    """Hard sigmoid function."""
    return np.clip((x + 1) / 2, 0, 1)

def logit(x):
    """Logit function with protection against division by zero."""
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def mod(x, y):
    """Modulo operation with protection against division by zero."""
    return np.mod(x, y + 1e-6)

def max_op(x, y):
    """Maximum of two values."""
    return np.maximum(x, y)

def min_op(x, y):
    """Minimum of two values."""
    return np.minimum(x, y)

def average(x, y):
    """Average of two values."""
    return (x + y) / 2











def f0(x: np.ndarray) -> np.ndarray:
    return add(x[0], mul(0.182591, min_op(x[1], 0.882941)))


def f1(x: np.ndarray) -> np.ndarray: 
    return custom_sin(x[0])


def f2(x: np.ndarray) -> np.ndarray:
    return safe_div(leaky_relu(leaky_relu(average(min_op(add(min_op(add(add(add(x[1], x[0]), x[0]), min_op(add(cube(safe_pow(safe_log(max_op(sub(1.47295, x[0]), 4.85972)), logit(triangle_wave(x[0])))), x[2]), min_op(max_op(add(triangle_wave(x[0]), x[2]), safe_pow(4.77956, -0.711423)), custom_exp(logit(safe_pow(logit(safe_pow(1.79359, elu(4.33868, mod(x[1], 3.65731)))), safe_pow(1.79359, elu(4.33868, mod(x[1], 3.65731))))))))), min_op(add(cube(x[0]), x[2]), min_op(max_op(leaky_relu(gaussian(4.61924), sub(x[1], triangle_wave(-2.89579))), 4.85972), 3.85772))), min_op(add(add(x[1], x[0]), x[2]), min_op(add(x[1], x[0]), sub(x[1], -3.998)))), cube(relu(safe_log(mul(4.65932, square_wave(safe_pow(triangle_wave(x[0]), relu(sinc(sub(1.47295, x[0])))))))))), 4.65932), relu(safe_log(sub(logit(elu(x[1], 2.39479)), add(add(add(x[1], x[0]), x[0]), min_op(add(cube(safe_pow(safe_log(max_op(sub(1.47295, x[0]), 4.85972)), logit(triangle_wave(x[0])))), x[2]), min_op(max_op(logit(x[2]), safe_pow(4.77956, -0.711423)), custom_exp(logit(safe_pow(logit(safe_pow(1.79359, elu(cube(x[0]), mod(x[1], 3.65731)))), safe_pow(1.79359, elu(4.33868, mod(x[1], 3.65731))))))))))))), relu(sinc(leaky_relu(x[0], 0.831663)))), mul(bent_identity(leaky_relu(gaussian(4.61924), sub(logit(elu(x[1], 2.39479)), triangle_wave(-2.89579)))), sub(triangle_wave(safe_pow(x[1], -4.23848)), leaky_relu(logit(elu(x[1], sinc(sub(1.47295, x[0])))), sub(logit(elu(x[1], 2.39479)), add(add(add(x[1], x[0]), x[0]), min_op(add(cube(safe_pow(safe_log(max_op(sub(1.47295, x[0]), 4.85972)), logit(triangle_wave(x[0])))), x[2]), min_op(max_op(logit(x[2]), safe_pow(4.77956, -0.711423)), custom_exp(logit(triangle_wave(x[0])))))))))))


def f3(x: np.ndarray) -> np.ndarray:
    return add(sub(square(mul(-1.12306, x[0])), cube(x[1])), square(min_op(add(safe_sqrt(x[2]), min_op(-1.95848, x[0])), min_op(min_op(x[2], add(x[2], min_op(add(safe_sqrt(x[2]), min_op(-1.95848, x[0])), sub(safe_sqrt(x[2]), x[0])))), sub(safe_sqrt(x[2]), x[0])))))


def f4(x: np.ndarray) -> np.ndarray:
    return mul(min_op(min_op(sinc(bent_identity(sub(0.350701, x[1]))), sinc(sub(bent_identity(sub(sinc(sub(leaky_relu(-4.0982, custom_sin(-1.35271)), x[1])), add(x[1], custom_exp(average(min_op(safe_sqrt(3.45691), x[0]), min_op(-4.97996, 2.09419)))))), elu(x[1], x[1])))), sinc(relu(max_op(max_op(bent_identity(min_op(sub(bent_identity(4.65932), elu(x[1], 0.350701)), x[1])), safe_log(-4.0982)), bent_identity(min_op(sub(leaky_relu(max_op(-1.91383, x[1]), custom_sin(elu(safe_div(x[1], average(x[1], 3.65731)), 2.05411))), x[1]), relu(max_op(max_op(sinc(-1.35271), safe_log(-4.0982)), min_op(softsign(x[0]), max_op(average(3.37675, x[0]), x[0])))))))))), bent_identity(bent_identity(bent_identity(4.65932))))


def f5(x: np.ndarray) -> np.ndarray:
    return relu(-0.771543)


def f6(x: np.ndarray) -> np.ndarray:
    return add(sub(x[1], x[0]), average(mul(safe_pow(-2.63527, gaussian(2.51503)), x[1]), average(average(x[1], average(x[1], x[0])), x[0])))


def f7(x: np.ndarray) -> np.ndarray:
    return mul(elu(leaky_relu(mul(x[0], mul(-4.0045, x[1])), -4.0045), x[1]), average(elu(elu(elu(mul(-3.8044, 4.75988), mul(x[0], mul(x[1], 1.15308))), mul(x[0], triangle_wave(mod(mul(mod(x[1], x[0]), 1.07804), x[0])))), elu(mul(-3.8044, 4.75988), mul(elu(mul(x[0], 4.75988), mul(x[0], triangle_wave(mod(mul(mod(x[1], x[0]), 1.07804), x[0])))), triangle_wave(mod(mul(mod(x[1], -4.0045), 1.07804), x[0]))))), mul(x[0], x[1])))
    
    
def f8(x: np.ndarray) -> np.ndarray:
    #return cube(safe_div(x[5], reciprocal(elu(x[5], min_op(min_op(-4.36968, mod(0.162581, add(x[4], -0.517759))), min_op(mod(0.162581, mod(0.162581, add(x[5], -0.517759))), x[4]))))))
    
    return mul(sub(sub(sub(sub(sub(average(safe_pow(leaky_relu(logit(triangle_wave(-0.270541)), safe_sqrt(4.01804)), safe_sqrt(x[5])), leaky_relu(logit(triangle_wave(safe_sqrt(x[5]))), x[5])), logit(safe_sqrt(average(sub(sub(sub(triangle_wave(safe_sqrt(x[5])), logit(safe_sqrt(average(max_op(x[1], 1.25251), logit(safe_sqrt(x[5])))))), logit(safe_sqrt(average(average(x[3], 4.01804), logit(safe_sqrt(x[5])))))), logit(x[5])), logit(sub(average(safe_pow(x[5], safe_sqrt(x[5])), leaky_relu(logit(max_op(x[1], 1.25251)), x[5])), logit(safe_sqrt(average(x[3], 4.01804))))))))), logit(safe_sqrt(average(sub(sub(logit(triangle_wave(safe_sqrt(x[5]))), logit(safe_sqrt(average(x[3], average(x[3], logit(triangle_wave(-0.270541))))))), logit(x[5])), logit(safe_sqrt(logit(safe_sqrt(average(max_op(x[1], 1.25251), logit(safe_sqrt(x[5]))))))))))), logit(x[5])), logit(safe_sqrt(average(sub(sub(sub(triangle_wave(-0.270541), logit(safe_sqrt(average(max_op(x[1], 1.25251), logit(safe_sqrt(x[5])))))), logit(safe_sqrt(average(max_op(x[1], 1.25251), logit(safe_sqrt(x[5])))))), logit(x[5])), logit(sub(average(safe_pow(x[5], safe_sqrt(x[5])), leaky_relu(logit(triangle_wave(safe_sqrt(x[5]))), x[5])), logit(safe_sqrt(average(x[3], 4.01804))))))))), logit(x[5])), mul(3.73747, x[5]))