from algorithm import parallelize, vectorize
from math import max, pow, exp

from ..structs import Matrix
from ..utils import tile


fn mse_loss(mse: Matrix, y_pred: Matrix, y_true: Matrix) -> None:
    """ Function `mse_loss`: calculates mean squared error between two
    identically-shaped matrices.

    Args:
        mse: (1, 1) matrix storing DType.float32 result (mse).
        y_pred: (m, n) matrix, predictions.
        y_true: (m, n) matrix, true values.
    """

    alias nelts = simdwidthof[DType.float32]()

    @parameter
    fn calc_row(m: Int):

        @parameter
        fn _mse[nelts: Int](n: Int):
            let sum_sq = pow[DType.float32, nelts](
                y_pred.load[nelts](m, n) - y_true.load[nelts](m, n), 2
            ).reduce_add()
            mse.store[1](0, 0, mse.load[1](0, 0) + sum_sq[0])

        vectorize[nelts, _mse](y_pred.cols)

    parallelize[calc_row](y_pred.rows, y_pred.rows)


fn relu(Y: Matrix, X: Matrix) -> None:
    """ Function `relu`: apply rectified linear activation to supplied matrix.

    Args:
        Y: Resulting matrix shape (m, n).
        X: Input matrix shape (m, n).
    """

    alias nelts = simdwidthof[DType.float32]()

    @parameter
    fn calc_row(m: Int):

        @parameter
        fn _relu[nelts: Int](n: Int):
            Y.store[nelts](m, n, max[DType.float32, nelts](
                X.load[nelts](m, n), 0.0
            ))

        vectorize[nelts, _relu](Y.cols)

    parallelize[calc_row](Y.rows, Y.rows)


fn sigmoid(Y: Matrix, X: Matrix) -> None:
    """ Function `sigmoid`: apply sigmoid activation to supplied matrix.

    Args:
        Y: Resulting matrix shape (m, n).
        X: Input matrix shape (m, n).
    """

    alias nelts = simdwidthof[DType.float32]()

    @parameter
    fn calc_row(m: Int):

        @parameter
        fn _sigmoid[nelts: Int](n: Int):
            let e_x = exp(X.load[nelts](m, n))
            Y.store[nelts](m, n, e_x / (1 + e_x))

        vectorize[nelts, _sigmoid](Y.cols)

    parallelize[calc_row](Y.rows, Y.rows)
