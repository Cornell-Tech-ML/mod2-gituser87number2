from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    @abstractmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    @abstractmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Creates apply handling for hidden backend scalar funcs"""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition"""
        return float(a + b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition"""
        return d_output, d_output


class LogScalar(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for log"""
        ctx.save_for_backward(a)
        return float(operators.log(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for log"""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function f(x, y) = x * y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication"""
        ctx.save_for_backward(a, b)
        return float(a * b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication"""
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function f(x) = 1 / x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse"""
        ctx.save_for_backward(a)
        return float(1 / a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inverse"""
        (a,) = ctx.saved_values
        return -d_output / (a**2)


class Neg(ScalarFunction):
    """Negation function f(x) = -x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation"""
        return float(-a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation"""
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function f(x) = 1 / (1 + exp(-x))"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid"""
        sigmoid_val = 1 / (1 + operators.exp(-a))
        ctx.save_for_backward(sigmoid_val)
        return float(sigmoid_val)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid"""
        (sigmoid_val,) = ctx.saved_values
        return d_output * sigmoid_val * (1 - sigmoid_val)


class Relu(ScalarFunction):
    """ReLU function f(x) = max(0, x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for relu"""
        ctx.save_for_backward(a)
        return float(max(0, a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for relu"""
        (a,) = ctx.saved_values
        return d_output if a > 0 else 0


class ExpScalar(ScalarFunction):
    """Exponential function f(x) = exp(x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exp"""
        exp_val = operators.exp(a)
        ctx.save_for_backward(exp_val)
        return float(exp_val)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exp"""
        (exp_val,) = ctx.saved_values
        return d_output * exp_val


class LT(ScalarFunction):
    """Less than function f(x, y) = 1 if x < y else 0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less than"""
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less than"""
        return 0.0, 0.0  # No gradient for binary comparison


class EQ(ScalarFunction):
    """Equality function f(x, y) = 1 if x == y else 0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equivalence comparison"""
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equivalence comparison"""
        return 0.0, 0.0  # No gradient for binary comparison
