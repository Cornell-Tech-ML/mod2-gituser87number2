from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals2 = list(vals)

    # Perturb the argument symmetrically by epsilon on both sides
    vals2[arg] += epsilon
    f_plus = f(*vals2)

    vals2[arg] -= 2 * epsilon  # Go from +epsilon to -epsilon
    f_minus = f(*vals2)

    # Reset the argument to its original value
    vals2[arg] += epsilon  # Back to the original value

    # Compute the central difference derivative approximation
    derivative = (f_plus - f_minus) / (2 * epsilon)

    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate derivative function"""

    ...

    @property
    def unique_id(self) -> int:
        """Returns the unique identifier for the variable.

        Returns
        -------
            int: The unique identifier.

        """
        ...

    def is_leaf(self) -> bool:
        """Checks if scalar input val is a leaf"""
        ...

    def is_constant(self) -> bool:
        """Checks if scalar input val is a constant"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns parents for given scalar"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Function for performing chain rule derivation"""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = {}
    order = []

    def dfs(v: Variable) -> None:
        """DFS Search method"""
        if v not in visited:
            visited[v] = True
            for parent in v.parents:
                dfs(parent)
            order.append(v)

    dfs(variable)

    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable.
        deriv: The derivative we want to propagate backward to the leaves.

    Returns:
    -------
        None: Updates the derivative values of each leaf through accumulate_derivative

    """
    deriv_dict = {variable.unique_id: deriv}

    sorted_vars = topological_sort(variable)  # Calls topo sort

    # Iterate through the topological order and calculate the derivatives
    for var in sorted_vars:
        if var.is_leaf():
            continue

        # Get the derivatives of the current variable
        varDer = var.chain_rule(deriv_dict[var.unique_id])

        # Accumulate the derivative for each parent of the current variable
        for parent_var, d_input in varDer:
            if parent_var.is_leaf():
                parent_var.accumulate_derivative(d_input)
            else:
                if parent_var.unique_id in deriv_dict:
                    deriv_dict[parent_var.unique_id] += d_input
                else:
                    deriv_dict[parent_var.unique_id] = d_input


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns saved values corresponding to self"""
        return self.saved_values
