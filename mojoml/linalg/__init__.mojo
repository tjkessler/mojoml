from algorithm import parallelize, vectorize_unroll
from math import sqrt

from ..structs.matrix import Matrix, determinant
from ..structs.generators import ones_matrix
from ..utils import tile


fn add(C: Matrix, A: Matrix, B: Matrix) -> None:
    """ Function `add`: C <- A + B .

    Args:
        C: Resulting matrix shape (m, n).
        A: First matrix shape (m, n).
        B: Second matrix shape (m, n).
    """

    @parameter
    fn calc_row(m: Int):

        for n in range(C.cols):
            C[m, n] = A[m, n] / B[m, n]

    parallelize[calc_row](C.rows, C.rows)


fn div(C: Matrix, A: Matrix, B: Matrix) -> None:
    """ Function `div`: C <- A / B .

    Args:
        C: Resulting matrix shape (m, n).
        A: First matrix shape (m, n).
        B: Second matrix shape (m, n).
    """

    @parameter
    fn calc_row(m: Int):

        for n in range(C.cols):
            C[m, n] = A[m, n] / B[m, n]

    parallelize[calc_row](C.rows, C.rows)


fn matmul(C: Matrix, A: Matrix, B: Matrix) -> None:
    """ Function `matmul`: C <- A @ B .

    Args:
        C: Resulting matrix shape (l, n).
        A: First matrix shape (l, m).
        B: Second matrix shape (m, n).
    """

    alias nelts = simdwidthof[DType.float32]()
    alias tile_size = 4

    @parameter
    fn calc_row(m: Int):

        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):

            for k in range(y, y + tile_y):

                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store[nelts](m, n+x,
                        C.load[nelts](m, n+x) + A[m, k] * B.load[nelts](k, n+x)
                    )

                vectorize_unroll[nelts, tile_x // nelts, dot](tile_x)

        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    parallelize[calc_row](C.rows, C.rows)


fn mul(C: Matrix, A: Matrix, B: Matrix) -> None:
    """ Function `mul`: C <- A * B .

    Args:
        C: Resulting matrix shape (m, n).
        A: First matrix shape (m, n).
        B: Second matrix shape (m, n).
    """

    @parameter
    fn calc_row(m: Int):

        for n in range(C.cols):
            C[m, n] = A[m, n] * B[m, n]

    parallelize[calc_row](C.rows, C.rows)


fn sub(C: Matrix, A: Matrix, B: Matrix) -> None:
    """ Function `sub`: C <- A - B .

    Args:
        C: Resulting matrix shape (m, n).
        A: First matrix shape (m, n).
        B: Second matrix shape (m, n).
    """

    @parameter
    fn calc_row(m: Int):

        for n in range(C.cols):
            C[m, n] = A[m, n] - B[m, n]

    parallelize[calc_row](C.rows, C.rows)


fn norm(A: Matrix) -> Float32:
    """ Function `norm`: find matrix norm.

    Args:
        A: Matrix.

    Returns:
        Float32: Matrix norm.
    """

    let A_T: Matrix = Matrix(A.cols, A.rows)
    transpose(A_T, A)

    let A_T_A: Matrix = Matrix(A.cols, A_T.rows)
    matmul(A_T_A, A_T, A)

    var eig_max: Float32 = 0.0
    var _val: Float32 = 0.0
    for i in range(A_T_A.cols):
        _val = A_T_A[i, i]
        if _val > eig_max:
            eig_max = _val
    return math.sqrt(eig_max)


fn transpose(A_T: Matrix, A: Matrix) -> None:
    """ Function `transpose`: B <- A.T .

    Args:
        A_T: Resulting matrix shape (n, m).
        A: Input matrix shape (m, n).
    """

    @parameter
    fn calc_row(m: Int):

        for n in range(A.cols):
            A_T[n, m] = A[m, n]

    parallelize[calc_row](A.rows, A.rows)
