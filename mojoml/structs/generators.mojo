

from .matrix import Matrix
from .vector import Vector


fn random_matrix(rows: Int, cols: Int) -> Matrix:
    """ Function `random_matrix`: creates a (rows, cols) shape Matrix with
    random values (from normal dist.).

    Args:
        rows: Matrix dim 0.
        cols: Matrix dim 1.

    Returns:
        Matrix: Random matrix shape (rows, cols).
    """

    let data = DTypePointer[DType.float32].alloc(rows * cols)
    random.rand(data, rows * cols)
    return Matrix(rows, cols, data)


fn random_vector(length: Int) -> Vector:
    """ Function `random_vector`: creates vector of specified length with
    random values (from normal dist.).

    Args:
        length: Length/size of vector.

    Returns:
        Vector: Random vector of `length`.
    """

    let data = DTypePointer[DType.float32].alloc(length)
    random.rand(data, length)
    return Vector(length, data)


fn identity_matrix(dim: Int) -> Matrix:
    """ Function `identity_matrix`: creates a (dim, dim) shape identity Matrix
    (values of 1.0 along diagonal).

    Args:
        dim: Dimension of identity matrix.

    Returns:
        Matrix: Identity matrix shape (dim, dim).
    """

    let data = DTypePointer[DType.float32].alloc(dim * dim)
    memset_zero(data, dim * dim)
    for i in range(dim):
        data.store(i * dim + i, 1.0)
    return Matrix(dim, dim, data)


fn ones_matrix(rows: Int, cols: Int) -> Matrix:
    """ Function `ones_matrix`: creates a (rows, cols) shape matrix with all
    entries = 1.0.

    Args:
        rows: Dim 0 of matrix.
        cols: Dim 1 of matrix.

    Returns:
        Matrix: Ones matrix shape (rows, cols).
    """

    let data = DTypePointer[DType.float32].alloc(rows * cols)
    memset_zero(data, rows * cols)
    for i in range(rows * cols):
        data.store(i, 1.0)
    return Matrix(rows, cols, data)
