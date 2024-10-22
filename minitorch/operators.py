"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable


# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul


def mul(x: float, y: float) -> float:
    """Multiplies x and y"""
    return x * y


# - id


def id(x: float) -> float:
    """Returns input unchanged"""
    return x


# - add


def add(x: float, y: float) -> float:
    """Adds x and y"""
    return x + y


# - neg


def neg(x: float) -> float:
    """Returns negation of input value"""
    return -1 * x


# - lt
def lt(x: float, y: float) -> bool:
    """Returns True if x is less than y, otherwise returns False"""
    if x < y:
        return True
    else:
        return False


# - eq


def eq(x: float, y: float) -> bool:
    """Returns True if the 2 input values are equivalent, otherwise returns False"""
    if x == y:
        return True
    else:
        return False


# - max


def max(x: float, y: float) -> float:
    """Returns the larger of two float inputs"""
    if x >= y:
        return x
    else:
        return y


# - is_close


def is_close(x: float, y: float) -> bool:
    """Returns True if 2 values are less than 0.01 apart, otherwise False"""
    if abs(x - y) < 0.01:
        return True
    else:
        return False


# - sigmoid


def sigmoid(x: float) -> float:
    """Returns sigmoid function output of input value"""
    if x >= 0:
        sig = 1.0 / (1 + math.exp(-x))
    else:
        sig = math.exp(x) / (1 + math.exp(x))

    return sig


# - relu


def relu(x: float) -> float:
    """Returns input value if positive, otherwise returns 0"""
    val = max(0, x)
    return val


# - log


def log(x: float) -> float:
    """Computes logarithm of input with respect to e and returns value"""
    return math.log(x + 1e-10)


# - exp


def exp(x: float) -> float:
    """Computes value of e^input and returns value"""
    return math.exp(x)


# - log_back


def log_back(x: float, y: float) -> float:
    """Computes the derivative of the logarithm of x with respect to e and returns value multiplied with y"""
    return y * (1.0 / (x + 1e-10))


# - inv


def inv(x: float) -> float:
    """Returns inverse of input value"""
    return 1.0 / x


# - inv_back


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of the inverse of x and returns value multiplied with y"""
    return -y * (x ** (-2))


# - relu_back


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU of x and returns value multiplied with y"""
    if x > 0:
        return y
    else:
        return 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map


def map(func: Callable, lin: list[float]) -> list[float]:
    """Higher-order function that applies a given function to each element of an iterable

    Args:
    ----
        func: function
        lin: list of floats

        returns: list of floats

    """
    length = len(lin)

    if length == 0:
        return []

    newList = [0.0] * length
    for i in range(length):
        newList[i] = func(lin[i])

    return newList


# - zipWith


def zipWith(func: Callable, list1: list[float], list2: list[float]) -> list[float]:
    """Higher-order function that combines elements from two iterables using a given function

    Args:
    ----
    func: function
    list1: list of floats
    list2: list of floats

    returns: list of floats

    """
    length = len(list1)

    if list1 == []:
        return []

    newList = [0.0] * length
    for i in range(length):
        newList[i] = func(list1[i], list2[i])

    return newList


# - reduce


def reduce(func: Callable, input1: Iterable[float]) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function.
        func (Callable): A function that takes two floats and returns a float.
        input1 (Iterable[float]): An iterable of floats to be reduced.

    Returns
    -------
        float: The reduced single value.

    """
    input1 = list(input1)
    length = len(input1)

    if length == 1:
        return input1[0]
    elif length == 0:
        return 0.0

    val: float = func(input1[0], input1[1])
    for i in range(length - 2):
        val = func(val, input1[i + 2])

    return val


#
# Use these to implement
# - negList : negate a list


def negList(list1: list[float]) -> list[float]:
    """Negates all elements of a list using map

    Args:
    ----
        list1: list of floats

        returns: list of floats

    """
    return map(neg, list1)


# - addLists : add two lists together


def addLists(list1: list[float], list2: list[float]) -> list[float]:
    """Adds two lists elementwise

    Args:
    ----
        list1: list of floats
        list2: list of floats

        returns: list of floats

    """
    return zipWith(add, list1, list2)


# - sum: sum lists


def sum(list1: Iterable[float]) -> float:
    """Adds each element in an iterable and computes a final result

    Args:
    ----
    list1: list of floats

    returns: list of floats

    """
    return reduce(add, list1)


# - prod: take the product of lists


def prod(input1: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce

    Args:
    ----
    input1: Iterable of floats

    returns: float

    """
    return reduce(mul, input1)


# TODO: Implement for Task 0.3.
