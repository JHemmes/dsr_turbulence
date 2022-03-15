"""Class for symbolic expression object or program."""

import array
import os
import platform
import warnings
from textwrap import indent

import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy import pretty
from functools import lru_cache

from dsr.functions import PlaceholderConstant, AD_PlaceholderConstant
from dsr.const import make_const_optimizer
from dsr.utils import cached_property
import dsr.utils as U
from dsr.library import Token
from copy import deepcopy, copy


def _finish_tokens(tokens):
    """
    Complete the pre-order traversal. using secondary library!

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal.

    Returns
    _______
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    """
    try:
        arities = np.array([Program.sec_library.arities[t] for t in tokens])
    except IndexError:
        print('pause_here')
    dangling = 1 + np.cumsum(arities - 1)

    if 0 in dangling:
        expr_length = 1 + np.argmax(dangling == 0)
        tokens = tokens[:expr_length]
    else:
        # Extend with constants until complete
        tokens = np.append(tokens, np.full(dangling[-1], Program.sec_library.names.index('const')))

    return tokens

def find_mul_token():
    return Program.library.names.index('mul')

def find_add_token():
    return Program.library.names.index('add')


def from_str_tokens(str_tokens, optimize, skip_cache=False):
    """
    Memoized function to generate a Program from a list of str and/or float.
    See from_tokens() for details.

    Parameters
    ----------
    str_tokens : str | list of (str | float)
        Either a comma-separated string of tokens and/or floats, or a list of
        str and/or floats.

    optimize : bool
        See from_tokens().

    skip_cache : bool
        See from_tokens().

    Returns
    -------
    program : Program
        See from_tokens().
    """

    # Convert str to list of str
    if isinstance(str_tokens, str):
        str_tokens = str_tokens.split(",")

    # Convert list of str|float to list of tokens
    if isinstance(str_tokens, list):
        traversal = []
        constants = []
        for s in str_tokens:
            if s in Program.library.names:
                t = Program.library.names.index(s.lower())
            elif U.is_float(s):
                assert "const" not in str_tokens, "Currently does not support both placeholder and hard-coded constants."
                assert not optimize, "Currently does not support optimization with hard-coded constants."
                t = Program.library.const_token
                constants.append(float(s))
            else:
                raise ValueError("Did not recognize token {}.".format(s))
            traversal.append(t)
        traversal = np.array(traversal, dtype=np.int32)
    else:
        raise ValueError("Input must be list or string.")

    # Generate base Program (with "const" for constants)
    p = from_tokens(traversal, optimize=optimize, skip_cache=skip_cache)

    # Replace any constants
    p.set_constants(constants)

    return p


def from_tokens(tokens, optimize=False, skip_cache=False):
    """
    Memoized function to generate a Program from a list of tokens.

    Since some tokens are nonfunctional, this first computes the corresponding
    traversal. If that traversal exists in the cache, the corresponding Program
    is returned. Otherwise, a new Program is returned.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    optimize : dict or False
        if it is a dict, this dict contains the optimisation option
        if it is False no constant optimisation is carried out.

    skip_cache : bool
        Whether to bypass the cache when creating the program.

    Returns
    _______
    program : Program
        The Program corresponding to the tokens, either pulled from memoization
        or generated from scratch.
    """

    '''
        Truncate expressions that complete early; extend ones that don't complete
    '''
    if len(tokens.shape) > 1:
        # enforce sum of four tensors.

        n_tensors = tokens.shape[1]
        mul_token = find_mul_token()
        add_token = find_add_token()
        for ii in range(tokens.shape[1]):
            subtokens = _finish_tokens(tokens[:,ii])
            subtokens += n_tensors  # offset the integers by the number of tokens
            if ii == 0:
                subtokens = np.insert(subtokens, 0, [mul_token, ii])
                final_tokens = subtokens
            else:
                subtokens = np.insert(subtokens, 0, [add_token, mul_token, ii])
                final_tokens = np.insert(final_tokens, 0, subtokens)

        if skip_cache:
            p = Program(final_tokens, optimize=optimize)
        else:
            key = final_tokens.tostring()
            if key in Program.cache:
                p = Program.cache[key]
                p.count += 1
            else:
                p = Program(final_tokens, optimize=optimize)
                Program.cache[key] = p

    else:
        # create program as usual.

        tokens = _finish_tokens(tokens)
        # For stochastic Tasks, there is no cache; always generate a new Program.
        # For deterministic Programs, if the Program is in the cache, return it;
        # otherwise, create a new one and add it to the cache.
        if skip_cache:
            p = Program(tokens, optimize=optimize)
        elif Program.task.stochastic:
            p = Program(tokens, optimize=optimize)
        else:
            key = tokens.tostring()
            if key in Program.cache:
                p = Program.cache[key]
                p.count += 1
            else:
                p = Program(tokens, optimize=optimize)
                Program.cache[key] = p

    return p


class Program(object):
    """
    The executable program representing the symbolic expression.

    The program comprises unary/binary operators, constant placeholders
    (to-be-optimized), input variables, and hard-coded constants.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. "Dangling"
        programs are completed with repeated "x1" until the expression
        completes.

    optimize : bool
        Whether to optimize the program upon initializing it.

    Attributes
    ----------
    traversal : list
        List of operators (type: Function) and terminals (type: int, float, or
        str ("const")) encoding the pre-order traversal of the expression tree.

    tokens : np.ndarry (dtype: int)
        Array of integers whose values correspond to indices

    const_pos : list of int
        A list of indicies of constant placeholders along the traversal.

    float_pos : list of float
        A list of indices of constants placeholders or floating-point constants
        along the traversal.

    sympy_expr : str
        The (lazily calculated) SymPy expression corresponding to the program.
        Used for pretty printing _only_.

    base_r : float
        The base reward (reward without penalty) of the program on the training
        data.

    token_occurences : list
        List containing the number of occurences of each token.

    n_unique_tokens : float
        The number of unique tokens in the program

    complexity : float
        The (lazily calcualted) complexity of the program.

    r : float
        The (lazily calculated) reward of the program on the training data.

    count : int
        The number of times this Program has been sampled.

    str : str
        String representation of tokens. Useful as unique identifier.
    """

    # Static variables
    task = None             # Task
    library = None          # Library
    const_optimizer = None  # Function to optimize constants
    cache = {}

    # Cython-related static variables
    have_cython = None      # Do we have cython installed
    execute = None          # Link to execute. Either cython or python
    cyfunc = None           # Link to cyfunc lib since we do an include inline

    def __init__(self, tokens, optimize):

        """
        Builds the Program from a list of Tokens, optimizes the Constants
        against reward function, and evalutes the reward.
        """

        self.nfev = 0
        self.nit = 0
        self.top_quantile = 0
        self.ad_r = None
        self.traversal = [copy(Program.library[t]) for t in tokens]
        # self.lowmemory_traversal = [Program.library[t] for t in tokens]
        # self.traversal = [deepcopy(token) for token in self.lowmemory_traversal]
        self.const_pos = [i for i, t in enumerate(tokens) if Program.library[t].name == "const"]  # Just constant placeholder positions
        self.len_traversal = len(self.traversal)
        self.tokens = tokens
        self.invalid_tokens = None
        self.invalid = 0
        self.n_unique_tokens = len(np.unique(tokens))
        self.token_occurences = [np.sum(tokens == ii) for ii in range(self.library.L)]

        if self.have_cython and self.len_traversal > 1:
            self.is_input_var = array.array('i', [t.input_var is not None for t in self.traversal])

        self.str = tokens.tostring()

        if optimize: # optimise is a dict or False. If it is a dict it contains the optimiser options.
            _ = self.optimize(optim_opt=optimize)

        self.count = 1

    def cython_execute(self, X):
        """Executes the program according to X using Cython.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.
        """

        if self.len_traversal > 1:
            return self.cyfunc.execute(X, self.len_traversal, self.traversal, self.is_input_var)
        else:
            return self.python_execute(X)

    def python_execute(self, X):
        """Executes the program according to X using Python.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.
        """

        # # Check for single-node programs
        # node = self.traversal[0]
        # if isinstance(node, float):
        #     return np.repeat(node, X.shape[0])
        # if isinstance(node, int):
        #     return X[:, node]

        apply_stack = []
        counter = 0

        for node in self.traversal:

            node.index = counter
            counter += 1
            apply_stack.append([node])
            # while length of last entry = the arity + 1 of the last entry
            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                token = apply_stack[-1][0]
                terminals = apply_stack[-1][1:]
                # terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                #              else X[:, t] if isinstance(t, int)
                #              else t for t in apply_stack[-1][1:]]
                if token.input_var is not None:
                    intermediate_result = X[:, token.input_var]
                else:
                    globals()['idx_counter'] = token.index
                    intermediate_result = token(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        assert False, "Function should never get here!"
        return None


    def ad_python_execute(self, X):
        """Executes the program according to X using Python.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.
        """

        # # Check for single-node programs
        # node = self.traversal[0]
        # if isinstance(node, float):
        #     return np.repeat(node, X.shape[0])
        # if isinstance(node, int):
        #     return X[:, node]

        apply_stack = []
        counter = 0

        # FWD pass
        for node in self.ad_traversal:

            node.adjoint_val = 0
            node.index = counter
            counter += 1

            apply_stack.append([node])
            # while length of last entry = the arity + 1 of the last entry
            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                token = apply_stack[-1][0]
                terminals = apply_stack[-1][1:]

                if token.input_var is not None:
                    token.value = X[:, token.input_var]
                else:
                    globals()['idx_counter'] = token.index
                    token.value = token(*terminals)
                    # token.index = globals()['idx_counter']
                    # globals()['idx_counter'] += 1
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(token)
                else:
                    r = token.value
                    break

        # RWD pass:
        # set first adjoint value to 1
        self.ad_traversal[0].adjoint_val = 1

        # execute RWD pass, only update tokens that are parents of constants
        for token in self.ad_traversal:
            if token.parent_of_const:
                if token.name == 'const':
                    token.adjoint_val = np.sum(token.adjoint_val)
                else:
                    token.ad_function(token)

        # extract the const adjoint values from tree
        jacobian = []
        for ii in self.ad_const_pos:
            token = self.ad_traversal[ii]
            jacobian.append(token.adjoint_val)

        return r[0], np.array(jacobian)


    def optimize(self, optim_opt):
        """
        Optimizes the constant tokens against the training data and returns the
        optimized constants.

        This function generates an objective function based on the training
        dataset, reward function, and constant optimizer. It ignores penalties
        because the Program structure is fixed, thus penalties are all the same.
        It then optimizes the constants of the program and returns the optimized
        constants.

        Returns
        _______
        optimized_constants : vector
            Array of optimized constants.
        """

        def reverse_ad(consts):
            # if self.invalid > 0:
            #     self.reset_tokens_valid()
            self.invalid = False
            self.set_constants(consts, ad=True)
            self.ad_r, self.jac = self.task.ad_reverse(self)
            return -self.ad_r, -self.jac

        assert self.execute is not None, "set_execute needs to be called first"

        if len(self.const_pos) > 0:
            # Do the optimization

            # set ad_traversal
            self.task.set_ad_traversal(self)

            if self.traversal[self.const_pos[0]].value:
                # if this is the sub batch optimisation with no iter limit, set initial guess to be current constants
                # gtol = 1e-5
                x0 = np.zeros(len(self.const_pos))
                for ii in range(len(self.const_pos)):
                    x0[ii] = self.traversal[self.const_pos[ii]].value
            else:
                x0 = np.ones(len(self.const_pos))  # Initial guess

            # optimized_constants, nfev, nit = Program.const_optimizer(reverse_ad, x0, jac=True, options={'maxiter': maxiter})
            optimized_constants, nfev, nit = Program.const_optimizer(reverse_ad, x0, jac=True, options=optim_opt)

            self.nfev += nfev
            self.nit += nit

            # some times minimize returns nan constants, rendering the program invalid.
            if any(np.isnan(optimized_constants)):
                self.invalid = True
            self.set_constants(optimized_constants)

            # # delete optimisation variables to save cache memory
            self.ad_const_pos = self.jac = None
            self.ad_traversal = np.array([0, 0])  # set dummy array to not cause an error with invalid_log. update in self.evaluate()
        else:
            # No need to optimize if there are no constants
            optimized_constants = []

        return optimized_constants

    def reset_tokens_valid(self):
        for token in self.ad_traversal:
            token.invalid = False

    def replace_traversal(self):

        traversal = [Program.library[t] for t in self.tokens]
        if len(self.const_pos) > 0:
            consts = []
            for ii in self.const_pos:
                traversal[ii] = self.traversal[ii]
        self.traversal = traversal



    def set_constants(self, consts, ad=False):
        """Sets the program's constants to the given values"""

        for i, const in enumerate(consts):
            # Create a new instance of PlaceholderConstant instead of changing
            # the "values" attribute, otherwise all Programs will have the same
            # instance and just overwrite each other's value.
            if ad:
                const_to_append = AD_PlaceholderConstant(const)
                const_to_append.parent_of_const = True
                self.ad_traversal[self.ad_const_pos[i]] = const_to_append
            else:
                self.traversal[self.const_pos[i]] = PlaceholderConstant(const)

    @classmethod
    def clear_cache(cls):
        """Clears the class' cache"""

        cls.cache = {}

    @classmethod
    def tidy_cache(cls, n_hof):
        """Reduces the size of the class' cache"""
        """only retains the n_hof best expressions in cache"""

        keys = list(cls.cache.keys())
        values = list(cls.cache.values())
        rewards = [program.r for program in values]
        ad_rewards = [program.ad_r if program.ad_r else program.r for program in values]
        if not ad_rewards == rewards:
            rewards = np.array(rewards)
            ad_rewards = np.array(ad_rewards)
            ad_rewards[rewards == ad_rewards] = 0
            rewards = ad_rewards

        hof_idx = np.argsort(rewards)[-n_hof:][::-1]

        cls.cache = {keys[ii]: values[ii] for ii in hof_idx}




    @classmethod
    def set_task(cls, task):
        """Sets the class' Task"""

        Program.task = task
        Program.library = task.library
        Program.sec_library = task.sec_library

    @classmethod
    def set_const_optimizer(cls, name, **kwargs):
        """Sets the class' constant optimizer"""

        const_optimizer = make_const_optimizer(name, **kwargs)
        Program.const_optimizer = const_optimizer

    @classmethod
    def set_complexity_penalty(cls, name, weight):
        """Sets the class' complexity penalty"""

        all_functions = {
            # No penalty
            None : lambda p : 0.0,

            # Length of tree
            "length" : lambda p : len(p)
        }

        assert name in all_functions, "Unrecognzied complexity penalty name"

        if weight == 0:
            Program.complexity_penalty = lambda p : 0.0
        else:
            Program.complexity_penalty = lambda p : weight * all_functions[name](p)

    @classmethod
    def set_execute(cls, protected, invalid_weight):
        """Sets which execute method to use"""

        """
        If cython ran, we will have a 'c' file generated. The dynamic libary can be 
        given different names, so it's not reliable for testing if cython ran.
        """

        if platform.system() == 'Windows':
            execute_function = Program.python_execute
            Program.have_cython = False
            ad_execute = Program.ad_python_execute
        else:
            cpath = os.path.join(os.path.dirname(__file__), 'cyfunc.c')
            if os.path.isfile(cpath):
                from .                  import cyfunc
                Program.cyfunc          = cyfunc
                if invalid_weight > 0:
                    execute_function = Program.python_execute
                else:
                    # if invalid tokens are not logged, use cython and set dummy ixd_counter
                    execute_function = Program.cython_execute
                    globals()['idx_counter'] = 1
                Program.have_cython     = True
                ad_execute              = Program.ad_python_execute
            else:
                execute_function        = Program.python_execute
                Program.have_cython     = False
                ad_execute              = Program.ad_python_execute

        if protected:
            Program.execute = execute_function
        else:

            class InvalidLog():
                """Log class to catch and record numpy warning messages"""

                def __init__(self):
                    # self.error_type = None # One of ['divide', 'overflow', 'underflow', 'invalid']
                    # self.error_node = None # E.g. 'exp', 'log', 'true_divide'
                    self.new_entry = False  # Flag for whether a warning has been encountered during a call to Program.execute()
                    self.invalid_list = []

                def write(self, message):
                    """This is called by numpy when encountering a warning"""

                    # idx_counter is declared globally so it can be used here
                    self.invalid_list.append(globals()['idx_counter'])
                    self.new_entry = True

                if invalid_weight > 0:
                    # choose update function depending on whether invalid tokens are logged.
                    def update(self, p):
                        """If a floating-point error was encountered, set Program.invalid
                        to True and record the error type and error node."""

                        # # set invalid tokens for each program
                        # if len(p.const_pos) > 0:
                        #     p.invalid_tokens = np.zeros(len(p.ad_traversal))
                        # else:
                        #     p.invalid_tokens = np.zeros(len(p.traversal))

                        if self.new_entry:
                            # if invalid token is logged, update which tokens are invalid.
                            invalid_indices = np.unique(self.invalid_list)
                            # p.invalid_tokens[invalid_indices] = 1
                            #
                            p.invalid = True  # set to true here, later change to number of invalids

                            # reset invalid log
                            self.new_entry = False
                            self.invalid_list = []
                            return invalid_indices

                        return None
                else:
                    def update(self, p):
                        """If a floating-point error was encountered, set Program.invalid
                        to True and record the error type and error node."""

                        if self.new_entry:
                            p.invalid = True

                            # reset invalid log (reset list to avoid large cache)
                            self.new_entry = False
                            self.invalid_list = []

                        return None

            invalid_log = InvalidLog()
            np.seterrcall(invalid_log)  # Tells numpy to call InvalidLog.write() when encountering a warning

            # Define closure for execute function
            def unsafe_execute(p, X):
                """This is a wrapper for execute_function. If a floating-point error
                would be hit, a warning is logged instead, p.invalid is set to True,
                and the appropriate nan/inf value is returned. It's up to the task's
                reward function to decide how to handle nans/infs."""

                with np.errstate(all='log'):
                    y = execute_function(p, X)
                    invalid_indices = invalid_log.update(p)
                    return y, invalid_indices

            # Define closure for execute function
            def unsafe_ad_reverse(p, X):
                """This is a wrapper for execute_function. If a floating-point error
                would be hit, a warning is logged instead, p.invalid is set to True,
                and the appropriate nan/inf value is returned. It's up to the task's
                reward function to decide how to handle nans/infs."""

                with np.errstate(all='log'):
                    y = ad_execute(p, X)
                    invalid_indices = invalid_log.update(p)
                    return y, invalid_indices

            Program.execute = unsafe_execute
            Program.ad_reverse = unsafe_ad_reverse

    @cached_property
    def complexity(self):
        """Evaluates and returns the complexity of the program"""

        return Program.complexity_penalty(self.traversal)


    @cached_property
    def base_r(self):
        """Evaluates and returns the base reward of the program on the training
        set"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if not self.ad_r is None:
                return self.ad_r
            else:
                return self.task.reward_function(self)

    @cached_property
    def r(self):
        """Evaluates and returns the reward of the program on the training
        set"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return self.base_r - self.complexity


    @cached_property
    def evaluate(self):
        """Evaluates and returns the evaluation metrics of the program."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # if self.ad_traversal is None:
            #     self.ad_traversal = np.array([0,0])  # set dummy ad_traversal so that invalid_log.update doesnt error.

            return self.task.evaluate(self)

    @cached_property
    def complexity_eureqa(self):
        """Computes sum of token complexity based on Eureqa complexity measures."""

        complexity = sum([t.complexity for t in self.traversal])
        return complexity


    @cached_property
    def sympy_expr(self):
        """
        Returns the attribute self.sympy_expr.

        This is actually a bit complicated because we have to go: traversal -->
        tree --> serialized tree --> SymPy expression
        """

        tree = self.traversal.copy()
        tree = build_tree(tree)
        tree = convert_to_sympy(tree)
        try:
            expr = parse_expr(tree.__repr__()) # SymPy expression
        except:
            expr = "N/A"

        return expr


    def pretty(self):
        """Returns pretty printed string of the program"""
        return pretty(self.sympy_expr)


    def print_stats(self):
        """Prints the statistics of the program"""
        print("\tReward: {}".format(self.r))
        try:
            print("\tFull dataset reward: {}".format(self.ad_r))
        except AttributeError:
            pass
        print("\tBase reward: {}".format(self.base_r))
        print("\tCount: {}".format(self.count))
        print("\tInvalid: {}".format(self.invalid))
        print("\tTraversal: {}".format(self))
        print("\tExpression:")
        print("{}\n".format(indent(self.pretty(), '\t  ')))


    def __repr__(self):
        """Prints the program's traversal"""

        return ','.join([repr(t) for t in self.traversal])


###############################################################################
# Everything below this line is currently only being used for pretty printing #
###############################################################################


# Possible library elements that sympy capitalizes
capital = ["add", "mul", "pow"]


class Node(object):
    """Basic tree class supporting printing"""

    def __init__(self, val):
        self.val = val
        self.children = []

    def __repr__(self):
        children_repr = ",".join(repr(child) for child in self.children)
        if len(self.children) == 0:
            return self.val # Avoids unnecessary parantheses, e.g. x1()
        return "{}({})".format(self.val, children_repr)


def build_tree(traversal):
    """Recursively builds tree from pre-order traversal"""

    op = traversal.pop(0)
    n_children = op.arity
    val = repr(op)
    if val in capital:
        val = val.capitalize()

    node = Node(val)

    for _ in range(n_children):
        node.children.append(build_tree(traversal))

    return node


def convert_to_sympy(node):
    """Adjusts trees to only use node values supported by sympy"""

    if node.val == "div":
        node.val = "Mul"
        new_right = Node("Pow")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "sub":
        node.val = "Add"
        new_right = Node("Mul")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "inv":
        node.val = Node("Pow")
        node.children.append(Node("-1"))

    elif node.val == "neg":
        node.val = Node("Mul")
        node.children.append(Node("-1"))

    elif node.val == "n2":
        node.val = "Pow"
        node.children.append(Node("2"))

    elif node.val == "n3":
        node.val = "Pow"
        node.children.append(Node("3"))

    elif node.val == "n4":
        node.val = "Pow"
        node.children.append(Node("4"))

    for child in node.children:
        convert_to_sympy(child)



    return node
