# examples\sin\sin_sgd.py

import argparse
import numpy as np
import matplotlib.pyplot as plt

import stage


def relu(x):
    return np.maximum(0.0, x)


def relu_grad(x):
    return (x > 0).astype(np.float64)


class Net:
    """
    1 -> hidden -> 1 のMLP
    sin(x)近似用
    """

    def __init__(
            self, 
            hidden=64, 
            seed=1
    ):
        np.random.seed(seed)

        self.w1 = np.random.randn(1, hidden) * 0.5
        self.b1 = np.zeros((1, hidden))

        self.w2 = np.random.randn(hidden, 1) * 0.5
        self.b2 = np.zeros((1, 1))

    def forward(self, x):

        z1 = x @ self.w1 + self.b1
        a1 = relu(z1)

        y = a1 @ self.w2 + self.b2

        cache = (x, z1, a1, y)
        return y, cache

    def backward(self, cache, y_true):

        x, z1, a1, y_pred = cache
        batch = x.shape[0]

        dy = 2 * (y_pred - y_true) / batch

        gw2 = a1.T @ dy
        gb2 = np.sum(dy, axis=0, keepdims=True)

        da1 = dy @ self.w2.T
        dz1 = da1 * relu_grad(z1)

        gw1 = x.T @ dz1
        gb1 = np.sum(dz1, axis=0, keepdims=True)

        return gw1, gb1, gw2, gb2

    def loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)


def train_model(
        args, 
        model, 
        x, 
        y
):

    opt_w1 = stage.optim.numpy.StageSGD(args.lr, args.tau, args.eps)
    opt_b1 = stage.optim.numpy.StageSGD(args.lr, args.tau, args.eps)
    opt_w2 = stage.optim.numpy.StageSGD(args.lr, args.tau, args.eps)
    opt_b2 = stage.optim.numpy.StageSGD(args.lr, args.tau, args.eps)

    losses = []

    for epoch in range(args.epochs):

        y_pred, cache = model.forward(x)

        loss = model.loss(y_pred, y)
        gw1, gb1, gw2, gb2 = model.backward(cache, y)

        opt_w1.update(model.w1, gw1)
        opt_b1.update(model.b1, gb1)
        opt_w2.update(model.w2, gw2)
        opt_b2.update(model.b2, gb2)

        losses.append(loss)

        if args.verbose and epoch % args.log_interval == 0:
            print(
                "Epoch {} Loss {:.6f}".format(
                    epoch,
                    loss
                )
            )

    return losses


def main():

    parser = argparse.ArgumentParser(description="STAGE NumPy sin approximation")

    parser.add_argument(
        "--epochs",
        type=int,
        default=3000,
        help="number of epochs (default: 3000)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning rate (default: 0.01)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="STAGE hyperparameter tau",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="STAGE hyperparameter eps",
    )

    parser.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="hidden layer size",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        help="logging interval",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show training logs",
    )

    args = parser.parse_args()

    # training data
    x = np.linspace(-2*np.pi, 2*np.pi, 256).reshape(-1, 1)
    y = np.sin(x)

    model = Net(hidden=args.hidden, seed=args.seed)

    losses = train_model(args, model, x, y)

    y_pred, _ = model.forward(x)

    print("final loss:", model.loss(y_pred, y))

if __name__ == "__main__":
    main()
