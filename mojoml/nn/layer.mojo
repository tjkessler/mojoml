from algorithm import vectorize
from utils.index import Index

from ..linalg import matmul, add
from ..structs import Matrix
from ..structs.generators import random_matrix


struct Linear:

    var weights: Matrix
    var biases: Matrix
    var in_dim: Int
    var out_dim: Int

    fn __init__(inout self: Self, in_dim: Int, out_dim: Int) -> None:
        """ Struct `Linear`: linear forward operator/layer, i.e.,
        Y = X @ W.T + B; initializes with random weights `W` and biases `B`.

        Args:
            in_dim: Layer input dimension.
            out_dim: Layer output dimension.
        """

        self.weights = random_matrix(in_dim, out_dim)
        self.biases = random_matrix(out_dim, 1)
        self.in_dim = in_dim
        self.out_dim = out_dim

    fn __init__(inout self: Self, weights: Matrix,
                biases: Matrix) raises -> None:
        """ Struct `Linear`: linear forward operator/layer, i.e.,
        Y = X @ W.T + B; uses supplied weights `W` and biases `B`.

        Args:
            weights: Layer weights shape (m, n).
            biases: Layer biases shape (n, 1).
        """

        if weights.cols != biases.rows:
            raise Error("Incompatible weights/biases shape")
        self.weights = Matrix(weights.rows, weights.cols, weights.data)
        self.biases = Matrix(biases.rows, biases.cols, biases.data)
        self.in_dim = weights.rows
        self.out_dim = weights.cols

    fn forward(self: Self, Y: Matrix, X: Matrix) -> None:
        """ Y = X @ W.T + B.

        Args:
            Y: Resulting matrix shape (m, out_dim).
            X: Input matrix shape (m, in_dim).
        """

        alias nelts = simdwidthof[DType.float32]()
        for i in range(X.rows):
            for j in range(X.cols):
                @parameter
                fn dot_bias[nelts: Int](k: Int):
                    Y.store[nelts](i, k, Y.load[nelts](i, k) + X[i, j] *\
                     self.weights.load[nelts](j, k) +\
                     self.biases.load[nelts](k, 0))
                vectorize[nelts, dot_bias](self.weights.cols)
