from .matrix import Matrix
from .vector import Vector


fn random_matrix(rows: Int, cols: Int) -> Matrix:

    let data = DTypePointer[DType.float32].alloc(rows * cols)
    random.rand(data, rows * cols)
    return Matrix(rows, cols, data)


fn random_vector(length: Int) -> Vector:

    let data = DTypePointer[DType.float32].alloc(length)
    random.rand(data, length)
    return Vector(length, data)


fn identity_matrix(dim: Int) -> Matrix:

    let data = DTypePointer[DType.float32].alloc(dim * dim)
    memset_zero(data, dim * dim)
    for i in range(dim):
        data.store(i * dim + i, 1.0)
    return Matrix(dim, dim, data)
