import random
from itertools import starmap, chain, pairwise
from operator import mul
from typing import Sequence, Iterator

from nanograd.nanograd import Value, Numeric


class Perceptron:
    def __init__(self, n_inputs: int):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self: "Perceptron", inputs: Sequence[Numeric]) -> Value:
        # noinspection PyArgumentList
        return sum(starmap(mul, zip(inputs, self.weights)), self.bias).tanh()

    def parameters(self) -> Iterator[Value]:
        return chain(self.weights, [self.bias])


class Layer:
    def __init__(self, n_inputs: int, n_perceptrons: int):
        self.perceptrons = [Perceptron(n_inputs) for _ in range(n_perceptrons)]

    def __call__(self, inputs: Sequence[Numeric]) -> Sequence[Value]:
        return [perceptron(inputs) for perceptron in self.perceptrons]

    def parameters(self) -> Iterator[Value]:
        return chain(*(perceptron.parameters() for perceptron in self.perceptrons))


class MultiLayerPerceptron:
    def __init__(self, n_inputs: int, n_outs: list[int]):
        self.layers = [
            Layer(n_inputs, n_perceptrons)
            for n_inputs, n_perceptrons in pairwise(chain([n_inputs], n_outs))
        ]

    def __call__(self, inputs: Sequence[Numeric]) -> Sequence[Numeric]:
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs

    def parameters(self) -> Iterator[Value]:
        return chain(*(layer.parameters() for layer in self.layers))
