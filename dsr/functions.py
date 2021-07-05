"""Common Tokens used for executable Programs."""

import numpy as np
from fractions import Fraction
from copy import deepcopy

from dsr.library import Token, AD_Token, PlaceholderConstant, AD_PlaceholderConstant

GAMMA = 0.57721566490153286060651209008240243104215933593992


"""Define custom unprotected operators"""
def logabs(x1):
    """Closure of log for non-positive arguments."""
    return np.log(np.abs(x1))

def expneg(x1):
    return np.exp(-x1)

def n3(x1):
    return np.power(x1, 3)

def n4(x1):
    return np.power(x1, 4)

def sigmoid(x1):
    return 1 / (1 + np.exp(-x1))

def harmonic(x1):
    if all(val.is_integer() for val in x1):
        return np.array([sum(Fraction(1, d) for d in range(1, int(val)+1)) for val in x1], dtype=np.float32)
    else:
        return GAMMA + np.log(x1) + 0.5/x1 - 1./(12*x1**2) + 1./(120*x1**4)

def ad_mul(self):
    # update left child adjoint value
    self.left_child.adjoint_val += self.adjoint_val*self.right_child.value
    # update right child adjoint value
    self.right_child.adjoint_val += self.adjoint_val*self.left_child.value

def ad_add(self):
    # update left child adjoint value
    self.left_child.adjoint_val += self.adjoint_val
    # update right child adjoint value
    self.right_child.adjoint_val += self.adjoint_val

def ad_sub(self):
    # update left child adjoint value
    self.left_child.adjoint_val += self.adjoint_val
    # update right child adjoint value
    self.right_child.adjoint_val += -self.adjoint_val

def ad_div(self):
    # update left child adjoint value
    self.left_child.adjoint_val += self.adjoint_val/self.right_child.value
    # update right child adjoint value
    self.right_child.adjoint_val += -self.adjoint_val*(self.value/self.right_child.value)

def ad_exp(self):
    # update left child adjoint value
    self.left_child.adjoint_val += self.adjoint_val*self.value

def ad_log(self):
    # update left child adjoint valuex
    self.left_child.adjoint_val += self.adjoint_val/self.left_child.value

def ad_sum(self):
    # update left child adjoint value
    self.left_child.adjoint_val += self.adjoint_val

def ad_n2(self):
    # update left child adjoint value
    self.left_child.adjoint_val += self.adjoint_val*2*self.left_child.value

def ad_sqrt(self):
    # update left child adjoint value
    self.left_child.adjoint_val += self.adjoint_val/(2*np.sqrt(self.left_child.value))




# Annotate unprotected ops
unprotected_ops = [
    # Binary operators
    Token(np.add, "add", arity=2, complexity=1),
    Token(np.subtract, "sub", arity=2, complexity=1),
    Token(np.multiply, "mul", arity=2, complexity=1),
    Token(np.divide, "div", arity=2, complexity=2),

    # Built-in unary operators
    Token(np.sin, "sin", arity=1, complexity=3),
    Token(np.cos, "cos", arity=1, complexity=3),
    Token(np.tan, "tan", arity=1, complexity=4),
    Token(np.exp, "exp", arity=1, complexity=4),
    Token(np.log, "log", arity=1, complexity=4),
    Token(np.sqrt, "sqrt", arity=1, complexity=4),
    Token(np.square, "n2", arity=1, complexity=2),
    Token(np.negative, "neg", arity=1, complexity=1),
    Token(np.abs, "abs", arity=1, complexity=2),
    Token(np.maximum, "max", arity=1, complexity=4),
    Token(np.minimum, "min", arity=1, complexity=4),
    Token(np.tanh, "tanh", arity=1, complexity=4),
    Token(np.reciprocal, "inv", arity=1, complexity=2),
    Token(np.sum, "sum", arity=1, complexity=2),

    # Custom unary operators
    Token(logabs, "logabs", arity=1, complexity=4),
    Token(expneg, "expneg", arity=1, complexity=4),
    Token(n3, "n3", arity=1, complexity=3),
    Token(n4, "n4", arity=1, complexity=3),
    Token(sigmoid, "sigmoid", arity=1, complexity=4),
    Token(harmonic, "harmonic", arity=1, complexity=4)
]

"""Define custom protected operators"""
def protected_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)

def protected_exp(x1):
    with np.errstate(over='ignore'):
        return np.where(x1 < 100, np.exp(x1), 0.0)

def protected_log(x1):
    """Closure of log for non-positive arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)

def protected_sqrt(x1):
    """Closure of sqrt for negative arguments."""
    return np.sqrt(np.abs(x1))

def protected_inv(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)

def protected_expneg(x1):
    with np.errstate(over='ignore'):
        return np.where(x1 > -100, np.exp(-x1), 0.0)

def protected_n2(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.square(x1), 0.0)

def protected_n3(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.power(x1, 3), 0.0)

def protected_n4(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.power(x1, 4), 0.0)

def protected_sigmoid(x1):
    return 1 / (1 + protected_expneg(x1))

# Annotate protected ops
protected_ops = [
    # Protected binary operators
    Token(protected_div, "div", arity=2, complexity=2),

    # Protected unary operators

    Token(protected_exp, "exp", arity=1, complexity=4),
    Token(protected_log, "log", arity=1, complexity=4),
    Token(protected_log, "logabs", arity=1, complexity=4), # Protected logabs is support, but redundant
    Token(protected_sqrt, "sqrt", arity=1, complexity=4),
    Token(protected_inv, "inv", arity=1, complexity=2),
    Token(protected_expneg, "expneg", arity=1, complexity=4),
    Token(protected_n2, "n2", arity=1, complexity=2),
    Token(protected_n3, "n3", arity=1, complexity=3),
    Token(protected_n4, "n4", arity=1, complexity=3),
    Token(protected_sigmoid, "sigmoid", arity=1, complexity=4)
]

# Add unprotected ops to function map
function_map = {
    op.name : op for op in unprotected_ops
    }

# Add protected ops to function map
function_map.update({
    "protected_{}".format(op.name) : op for op in protected_ops
    })

UNARY_TOKENS = set([op.name for op in function_map.values() if op.arity == 1])
BINARY_TOKENS = set([op.name for op in function_map.values() if op.arity == 2])


def create_tokens(n_input_var, function_set, protected):
    """
    Helper function to create Tokens.

    Parameters
    ----------
    n_input_var : int
        Number of input variable Tokens.

    function_set : list
        Names of registered Tokens, or floats that will create new Tokens.

    protected : bool
        Whether to use protected versions of registered Tokens.
    """

    tokens = []

    # Create input variable Tokens
    for i in range(n_input_var):
        token = Token(name="x{}".format(i + 1), arity=0, complexity=1,
                      function=None, input_var=i)
        tokens.append(token)

    for op in function_set:

        # Registered Token
        if op in function_map:
            # Overwrite available protected operators
            if protected and not op.startswith("protected_"):
                protected_op = "protected_{}".format(op)
                if protected_op in function_map:
                    op = protected_op

            token = function_map[op]

        # Hard-coded floating-point constant
        elif isinstance(op, float) or isinstance(op, int):
            name = str(op)
            value = np.atleast_1d(np.float32(op))
            function = lambda : value
            token = Token(name=name, arity=0, complexity=1, function=function)

        # Constant placeholder (to-be-optimized)
        elif op == "const":
            token = PlaceholderConstant()

        else:
            raise ValueError("Operation {} not recognized.".format(op))

        tokens.append(token)

    return tokens

# create dict of inputs to easily create ad tokens
ad_token_inputs = {'add':  {'function': np.add, 'ad_function': ad_add, 'name': 'add', 'arity': 2, 'complexity':1},
                   'sub':  {'function': np.subtract, 'ad_function': ad_sub, 'name': 'sub', 'arity': 2, 'complexity':1},
                   'mul':  {'function': np.multiply, 'ad_function': ad_mul, 'name': 'mul', 'arity': 2, 'complexity':1},
                   'div':  {'function': np.divide, 'ad_function': ad_div, 'name': 'div', 'arity': 2, 'complexity':2},
                   'exp':  {'function': np.exp, 'ad_function': ad_exp, 'name': 'exp', 'arity': 1, 'complexity':4},
                   'log':  {'function': np.log, 'ad_function': ad_log, 'name': 'log', 'arity': 1, 'complexity':4},
                   'sqrt': {'function': np.sqrt, 'ad_function': ad_sqrt, 'name': 'sqrt', 'arity': 1, 'complexity':4},
                   'n2':   {'function': np.square, 'ad_function': ad_n2, 'name': 'n2', 'arity': 1, 'complexity':2},
                   'sum':  {'function': np.sum, 'ad_function': ad_sum, 'name': 'sum', 'arity': 1, 'complexity':2}}

def create_ad_tokens(traversal):
    # must be unique tokens because otherwise the values of tokens appearing more than once are just overwritten

    ad_traversal = []

    for token in traversal:
        if token.input_var != None:
            token_instance = deepcopy(token)
            token_instance.left_child = None
            token_instance.right_child = None
            token_instance.value = None
            token_instance.adjoint_val = 0
            token_instance.index = None
            token_instance.parent_of_const = False
            ad_traversal.append(token_instance)
        elif token.name == 'const':
            ad_traversal.append(token)
        else:
            ad_traversal.append(AD_Token(**ad_token_inputs[token.name]))

    return ad_traversal

def create_metric_ad_tokens(names, y=None):
    # must be unique tokens because otherwise the values of tokens appearing more than once are just overwritten

    constant_dict = {'one': 1,
                     'y': y,
                     'n': len(y),
                     'n_var_y': np.var(y)*len(y)}

    metric_traversal = []

    for name in names:
        if name in constant_dict.keys():
            token = AD_PlaceholderConstant(value=constant_dict[name], name=name)
            metric_traversal.append(token)
        else:
            metric_traversal.append(AD_Token(**ad_token_inputs[name]))

    return metric_traversal