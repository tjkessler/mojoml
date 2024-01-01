from tensor import Tensor, TensorSpec
from utils.index import Index


fn minor(Y: Matrix, X: Matrix, row: Int, col: Int) -> None:
    """ Function `minor`: get minor matrix, i.e., excluding specified row
    and column.

    Args:
        Y: Resulting minor matrix shape (m - 1, n - 1).
        X: Input matrix shape (m, n).
        row: Row to exclude.
        col: Column to exclude.
    """

    var _r: Int = 0
    var _c: Int = 0
    for j in range(X.rows):
        if j != row:
            for i in range(X.cols):
                if i != col:
                    Y[_r, _c] = X[j, i]
                    _c += 1
            _r += 1
            _c = 0


struct Matrix:

    var data: DTypePointer[DType.float32]
    var rows: Int
    var cols: Int

    fn __init__(inout self: Self, rows: Int, cols: Int) -> None:
        """ Struct `Matrix`: matrix shape (rows, cols) of type `float32`.

        Args:
            rows: Matrix dim 0.
            cols: Matrix dim 1.
        """

        self.data = DTypePointer[DType.float32].alloc(rows * cols)
        memset_zero(self.data, rows * cols)
        self.rows = rows
        self.cols = cols

    fn __init__(inout self: Self, rows: Int, cols: Int,
                data: DTypePointer[DType.float32]) -> None:
        """ Struct `Matrix`: matrix shape (rows, cols) of type `float32`;
        initializes with supplied data.

        Args:
            rows: Matrix dim 0.
            cols: Matrix dim 1.
            data: Data used to construct matrix.
        """

        self.data = data
        self.rows = rows
        self.cols = cols

    fn repr(self: Self) -> String:

        var s: String = ""
        for j in range(self.rows):
            for i in range(self.cols):
                s += String(self.__getitem__(j, i))
                if i < self.cols - 1:
                    s += "\t"
            s += "\n"
        return s

    fn __getitem__(self: Self, y: Int, x: Int) -> Float32:
        """ Method `__getitem__`: get value at (y, x) with
        `Matrix[y, x]`.

        Args:
            y: Value dim 0.
            x: Value dim 1.

        Returns:
            Float32: Value at (y, x).
        """

        return self.load[1](y, x)

    fn __setitem__(self: Self, y: Int, x: Int, val: Float32) -> None:
        """ Method `__setitem__`: set value at (y, x) with
        `Matrix[y, x] = val`.

        Args:
            y: Value dim 0.
            x: Value dim 1.
            val: Value to set.
        """

        self.store[1](y, x, val)

    fn load[nelts: Int](self: Self, y: Int,
            x: Int) -> SIMD[DType.float32, nelts]:
        """ Method `load`: load SIMD-accelerated vector from data; loads
        `nelts` length vector starting at matrix coordinates (y, x); data
        loaded column-wise.

        Args:
            y: Init. coordinate dim 0.
            x: Init. coordinate dim 1.

        Returns:
            SIMD[DType.float32, nelts]: data in form of SIMD-vector.
        """

        return self.data.simd_load[nelts](y * self.cols + x)

    fn store[nelts: Int](self: Self, y: Int, x: Int,
             val: SIMD[DType.float32, nelts]) -> None:
        """ Method `store`: store SIMD-accelerated vector into Matrix data;
        stores `nelts` length vector starting at matrix coordinates (y, x);
        data stored column-wise.

        Args:
            y: Init. coordinate dim 0.
            x: Init. coordinate dim 1.
            val: SIMD-vector/data to store.
        """

        self.data.simd_store[nelts](y * self.cols + x, val)

    fn to_tensor(self: Self) -> Tensor[DType.float32]:
        """ Method `to_tensor`: converts matrix data to Mojo Tensor.

        Returns:
            Tensor[DType.float32]: Matrix data as Tensor object.
        """

        let spec = TensorSpec(DType.float32, self.rows, self.cols)
        var ten = Tensor[DType.float32](spec)
        for y in range(self.rows):
            for x in range(self.cols):
                ten[Index(y, x)] = self.__getitem__(y, x)
        return ten
