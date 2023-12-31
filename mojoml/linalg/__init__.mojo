from algorithm import parallelize, Static2DTileUnitFunc, vectorize_unroll

from ..structs.matrix import Matrix


fn norm(A: Matrix) -> Float32:

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


fn _tile[tiled_fn: Static2DTileUnitFunc, tile_x: Int, tile_y: Int](
         end_x: Int, end_y: Int) -> None:

    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


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

        _tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    parallelize[calc_row](C.rows, C.rows)


fn add(C: Matrix, A: Matrix, B: Matrix) raises -> None:
    """ Function `add`: C <- A + B .

    Args:
        C: Resulting matrix shape (m, n).
        A: First matrix shape (m, n).
        B: Second matrix shape (m, n).
    """

    alias nelts = simdwidthof[DType.float32]()
    alias tile_size = 4

    if A.rows != B.rows or A.cols != B.cols:
        raise Error("Invalid matrix sizes")

    @parameter
    fn calc_row(m: Int):

        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):

            for k in range(y, y + tile_y):

                @parameter
                fn _add[nelts: Int](n: Int):
                    C.store[nelts](m, n+x, A[m, k] + B.load[nelts](k, n+x))

                vectorize_unroll[nelts, tile_x // nelts, _add](tile_x)

        _tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    parallelize[calc_row](C.rows, C.rows)


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
