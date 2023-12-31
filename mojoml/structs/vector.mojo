from tensor import Tensor, TensorSpec
from utils.index import Index


struct Vector:

    var data: DTypePointer[DType.float32]
    var length: Int

    fn __init__(inout self: Self, length: Int) -> None:

        self.data = DTypePointer[DType.float32].alloc(length)
        memset_zero(self.data, length)
        self.length = length

    fn __init__(inout self: Self, length: Int,
                data: DTypePointer[DType.float32]) -> None:

        self.data = data
        self.length = length

    fn __getitem__(self: Self, i: Int) -> Float32:

        return self.load[1](i)

    fn __setitem__(self: Self, i: Int, val: Float32) -> None:

        self.store[1](i, val)

    fn load[nelts: Int](self: Self, i: Int) -> SIMD[DType.float32, nelts]:

        return self.data.simd_load[nelts](i)

    fn store[nelts: Int](self: Self, i: Int,
             val: SIMD[DType.float32, nelts]) -> None:

        self.data.simd_store[nelts](i, val)

    fn to_tensor(self: Self) -> Tensor[DType.float32]:

        let spec = TensorSpec(DType.float32, self.length)
        var ten = Tensor[DType.float32](spec)
        for i in range(self.length):
            ten[Index(i)] = self.__getitem__(i)
        return ten
