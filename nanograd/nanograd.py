import math
from dataclasses import dataclass, field
from functools import wraps
from graphlib import TopologicalSorter
from typing import Callable, Tuple, Union

from graphviz import Digraph

Numeric = Union[int, float, "Value"]
BackwardRuleBinary = Callable[["Value", "Value", "Value"], Tuple[float, float]]
BackwardRuleUnary = Callable[["Value", "Value"], float]
OpMethodBinary = Callable[["Value", Numeric], "Value"]
OpMethodUnary = Callable[["Value"], "Value"]


def autodiff_binary_op(
    backward_rule: BackwardRuleBinary,
) -> Callable[[OpMethodBinary], OpMethodBinary]:
    def decorator(op_method: OpMethodBinary) -> OpMethodBinary:
        @wraps(op_method)
        def wrapper(lhs: "Value", rhs: Numeric) -> "Value":
            out = op_method(lhs, rhs)
            lhs_converted, rhs_converted = out.parents

            def grad_fn() -> None:
                d_lhs, d_rhs = backward_rule(lhs_converted, rhs_converted, out)
                lhs_converted.grad += out.grad * d_lhs
                rhs_converted.grad += out.grad * d_rhs

            out.grad_fn = grad_fn

            return out

        return wrapper

    return decorator


def autodiff_unary_op(
    backward_rule: BackwardRuleUnary,
) -> Callable[[OpMethodUnary], OpMethodUnary]:
    def decorator(op_method: OpMethodUnary) -> OpMethodUnary:
        @wraps(op_method)
        def wrapper(self: "Value") -> "Value":
            out = op_method(self)
            (val_converted,) = out.parents

            def grad_fn() -> None:
                d_val = backward_rule(val_converted, out)
                val_converted.grad += out.grad * d_val

            out.grad_fn = grad_fn

            return out

        return wrapper

    return decorator


@dataclass(slots=True)
class Value:
    data: float
    name: str = ""
    op: str = ""
    parents: Tuple["Value", ...] = field(default_factory=tuple)
    grad_fn: Callable[[], None] = field(default_factory=lambda: lambda: None)
    grad: float = 0.0

    @autodiff_binary_op(lambda lhs, rhs, out: (1.0, 1.0))
    def __add__(self: "Value", other: Numeric) -> "Value":
        other_converted = Value._convert_value(
            other,
            f"unsupported operand type(s) for +: 'Value' and '{type(other).__name__}'",
        )

        return Value(
            data=self.data + other_converted.data,
            op="+",
            parents=(self, other_converted),
        )

    @autodiff_binary_op(lambda lhs, rhs, out: (rhs.data, lhs.data))
    def __mul__(self: "Value", other: Numeric) -> "Value":
        other_converted = Value._convert_value(
            other,
            f"unsupported operand type(s) for *: 'Value' and '{type(other).__name__}'",
        )

        return Value(
            data=self.data * other_converted.data,
            op="*",
            parents=(self, other_converted),
        )

    @autodiff_binary_op(
        lambda lhs, rhs, out: (1.0 / rhs.data, -lhs.data / (rhs.data ** 2))
    )
    def __truediv__(self: "Value", other: Numeric) -> "Value":
        other_converted = Value._convert_value(
            other,
            f"unsupported operand type(s) for /: 'Value' and '{type(other).__name__}'",
        )

        return Value(
            data=self.data / other_converted.data,
            op="/",
            parents=(self, other_converted),
        )

    def __rtruediv__(self: "Value", other: Numeric) -> "Value":
        other_converted = Value._convert_value(
            other,
            f"unsupported operand type(s) for /: '{type(other).__name__}' and 'Value'",
        )

        return other_converted / self

    @autodiff_unary_op(lambda lhs, out: 1.0 - out.data ** 2)
    def tanh(self: "Value") -> "Value":
        return Value(data=math.tanh(self.data), op="tanh", parents=(self,))

    @autodiff_unary_op(lambda lhs, out: out.data * (1.0 - out.data))
    def sigmoid(self: "Value") -> "Value":
        return Value(
            data=1.0 / (1.0 + math.exp(-self.data)), op="sigmoid", parents=(self,)
        )

    @autodiff_unary_op(lambda lhs, out: 1.0 / lhs.data)
    def ln(self: "Value") -> "Value":
        return Value(data=math.log(self.data), op="ln", parents=(self,))

    @autodiff_unary_op(lambda lhs, out: -1.0)
    def __neg__(self: "Value") -> "Value":
        return Value(
            data=-self.data,
            op="neg",
            parents=(self,),
        )

    def __sub__(self: "Value", other: Numeric) -> "Value":
        return self + (-other)

    def __rsub__(self: "Value", other: Numeric) -> "Value":
        other_converted = Value._convert_value(
            other,
            f"unsupported operand type(s) for -: '{type(other).__name__}' and 'Value'",
        )

        return other_converted - self

    @autodiff_binary_op(
        lambda lhs, rhs, out: (
            rhs.data * lhs.data ** (rhs.data - 1),
            out.data * math.log(lhs.data),
        )
    )
    def __pow__(self: "Value", other: Numeric) -> "Value":
        other_converted = Value._convert_value(
            other,
            f"unsupported operand type(s) for **: 'Value' and '{type(other).__name__}'",
        )

        return Value(
            data=self.data ** other_converted.data,
            op="**",
            parents=(self, other_converted),
        )

    def __rpow__(self: "Value", other: Numeric) -> "Value":
        other_converted = Value._convert_value(
            other,
            f"unsupported operand type(s) for **: '{type(other).__name__}' and 'Value'",
        )

        return other_converted ** self

    def backward(self: "Value") -> None:
        self.grad = 1.0
        graph = {node: node.parents for node in self._collect_all_nodes()}

        ts: TopologicalSorter = TopologicalSorter()

        for node, dependencies in graph.items():
            ts.add(node, *dependencies)

        topo_order = list(ts.static_order())

        for node in reversed(topo_order):
            node.grad_fn()

    def zero_grad(self: "Value") -> None:
        for node in self._collect_all_nodes():
            node.grad = 0.0

    def visualize(self: "Value") -> Digraph:
        dot: Digraph = Digraph()
        dot.attr(rankdir="LR", concentrate="true")
        nodes = {node: str(idx) for idx, node in enumerate(self._collect_all_nodes())}

        for node, node_id in nodes.items():
            label = f"data={node.data:.2f} | grad={node.grad:.2f}"

            if node.name:
                label = f"{node.name} |" + label

            dot.node(node_id, label=label, shape="record")

            if node.op:
                op_node_id = f"{node_id}_op"
                dot.node(op_node_id, label=node.op, shape="circle")

                for parent in node.parents:
                    parent_id = nodes[parent]
                    dot.edge(parent_id, op_node_id)

                dot.edge(op_node_id, node_id)
            else:
                for parent in node.parents:
                    parent_id = nodes[parent]
                    dot.edge(parent_id, node_id)

        return dot

    def _collect_all_nodes(self: "Value") -> set["Value"]:
        visited = set()
        stack = [self]

        while stack:
            value = stack.pop()
            if value not in visited:
                visited.add(value)
                stack.extend(value.parents)

        return visited

    @staticmethod
    def _convert_value(other: Numeric, error_message: str) -> "Value":
        if isinstance(other, (int, float)):
            return Value(data=other)
        elif not isinstance(other, Value):
            raise TypeError(error_message)
        else:
            return other

    __radd__ = __add__
    __rmul__ = __mul__
    __hash__ = object.__hash__
