from tensor import Tensor, TensorSpec
from utils.index import Index


struct Vector:

    var data: DTypePointer[DType.float32]
    var length: Int

    fn __init__(inout self: Self, length: Int) -> None:
        """ Struct `Vector`: vector of specified length type `float32`.

        Args:
            length: Length of the vector.
        """

        self.data = DTypePointer[DType.float32].alloc(length)
        memset_zero(self.data, length)
        self.length = length

    fn __init__(inout self: Self, length: Int,
                data: DTypePointer[DType.float32]) -> None:
        """ Struct `Vector`: vector of specified length type `float32`;
        initializes with supplied data.

        Args:
            length: Length of the vector.
            data: Data used to construct vector.
        """

        self.data = data
        self.length = length

    fn __getitem__(self: Self, i: Int) -> Float32:
        """ Method `__getitem__`: get value at specified index with
        `Vector[i]`.

        Args:
            i: Index of value.

        Returns:
            Float32: Value at index i.
        """

        return self.load[1](i)

    fn __setitem__(self: Self, i: Int, val: Float32) -> None:
        """ Method `__setitem__`: set value at specified index with
        `Vector[i] = val`.

        Args:
            i: Index of value.
            val: Value to set.
        """

        self.store[1](i, val)

    fn load[nelts: Int](self: Self, i: Int) -> SIMD[DType.float32, nelts]:
        """ Method `load`: load SIMD-accelerated sub-vector from data; loads
        `nelts` length vector starting at vector index `i`.

        Args:
            i: Init. index of load.

        Returns:
            SIMD[DType.float32, nelts]: data in form of SIMD-vector.
        """

        return self.data.simd_load[nelts](i)

    fn store[nelts: Int](self: Self, i: Int,
             val: SIMD[DType.float32, nelts]) -> None:
        """ Method `store`: store SIMD-accelerated sub-vector into vector data;
        stores `nelts` length vector starting at vector index `i`.

        Args:
            i: Init. index of store.
            val: SIMD-vector/data to store.
        """

        self.data.simd_store[nelts](i, val)

    fn to_tensor(self: Self) -> Tensor[DType.float32]:
        """ Method `to_tensor`: converts vector data to Mojo Tensor.

        Returns:
            Tensor[DType.float32]: Vector data as Tensor object.
        """

        let spec = TensorSpec(DType.float32, self.length)
        var ten = Tensor[DType.float32](spec)
        for i in range(self.length):
            ten[Index(i)] = self.__getitem__(i)
        return ten
