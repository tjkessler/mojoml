# mojoml 🔥

Linear algebra and machine learning in Mojo 🔥

(Heavily inspired by the [official Mojo documentation](https://docs.modular.com/mojo/))

## Usage

Move the `mojoml.mojopkg` file to your current working directory, or build from source with:

```
$ git clone https://github.com/tjkessler/mojoml
$ cd mojoml
$ mojo package mojoml -o mojoml.mojopkg
```

Basic usage (more to come!):


```python
from mojoml.structs import Matrix
from mojoml.structs.generators import random_matrix
from mojoml.linalg import matmul, transpose


fn main() -> None:

    let m1: Matrix = random_matrix(512, 512)
    let m2: Matrix = random_matrix(512, 512)
    let m3: Matrix = Matrix(512, 512)
    matmul(m3, m1, m2)  # m3 <- m1 @ m2
    let m4: Matrix = Matrix(512, 512)
    transpose(m4, m3)  # m4 <- m3.T
    var tensor_1: Tensor[DType.float32] = m4.to_tensor()

```
