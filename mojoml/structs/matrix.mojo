from tensor import Tensor, TensorSpec
from utils.index import Index


struct Matrix:

    var data: DTypePointer[DType.float32]
    var rows: Int
    var cols: Int

    fn __init__(inout self: Self, rows: Int, cols: Int) -> None:

        self.data = DTypePointer[DType.float32].alloc(rows * cols)
        memset_zero(self.data, rows * cols)
        self.rows = rows
        self.cols = cols

    fn __init__(inout self: Self, rows: Int, cols: Int,
                data: DTypePointer[DType.float32]) -> None:

        self.data = data
        self.rows = rows
        self.cols = cols

    fn __getitem__(self: Self, y: Int, x: Int) -> Float32:

        return self.load[1](y, x)

    fn __setitem__(self: Self, y: Int, x: Int, val: Float32) -> None:

        self.store[1](y, x, val)

    fn load[nelts: Int](self: Self, y: Int,
            x: Int) -> SIMD[DType.float32, nelts]:

        return self.data.simd_load[nelts](y * self.cols + x)

    fn store[nelts: Int](self: Self, y: Int, x: Int,
             val: SIMD[DType.float32, nelts]) -> None:

        self.data.simd_store[nelts](y * self.cols + x, val)

    fn to_tensor(self: Self) -> Tensor[DType.float32]:

        let spec = TensorSpec(DType.float32, self.rows, self.cols)
        var ten = Tensor[DType.float32](spec)
        for y in range(self.rows):
            for x in range(self.cols):
                ten[Index(y, x)] = self.__getitem__(y, x)
        return ten
