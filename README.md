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

Matrix operations:

```python
from mojoml.structs import Matrix
from mojoml.structs.generators import random_matrix
from mojoml.linalg import matmul, norm, transpose


fn main() -> None:

    let m1: Matrix = random_matrix(512, 512)
    let m2: Matrix = random_matrix(512, 512)

    # m_matmul <- m1 @ m2
    let m_matmul: Matrix = Matrix(512, 512)
    matmul(m_matmul, m1, m2)

    let m1_norm: Float32 = norm(m1)

    # m1_T <- m1.T
    let m1_T: Matrix = Matrix(512, 512)
    transpose(m1_T, m1)
```

Feed-forward, activation, loss (more to come!):

```python
from mojoml.structs import Matrix
from mojoml.structs.generators import random_matrix
from mojoml.nn import Linear
from mojoml.nn.functional import mse_loss, relu, sigmoid


fn main() -> None:

    # define a layer with 16 inputs, 32 outputs
    let layer: Linear = Linear(16, 32)

    # data to feed forward; 8 samples, 16 features per sample
    let inputs: Matrix = random_matrix(8, 16)

    # output shape is 8 samples, 32 outputs per sample
    let outputs: Matrix = Matrix(8, 32)

    # Y <- X @ W.T + B
    layer.forward(outputs, inputs)

    # apply ReLU activation
    let out_relu: Matrix = Matrix(8, 32)
    relu(out_relu, outputs)

    # apply sigmoid activation
    let out_sigmoid: Matrix = Matrix(8, 32)
    sigmoid(out_sigmoid, outputs)

    # calculate MSE loss w/ dummy target values
    let targets: Matrix = random_matrix(8, 32)
    let loss_mat: Matrix = Matrix(1, 1)
    mse_loss(loss_mat, out_sigmoid, targets)
    let mse: Float32 = loss_mat[0, 0]

    # TODO: gradient descent!
```
