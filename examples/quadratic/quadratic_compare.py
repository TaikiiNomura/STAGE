# examples\quadratic\quadratic_compare.py

import argparse
import numpy as np

import stage


class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    def update(self, params, grads):
        params -= self.lr * grads


def f(x):
    return (x - 3.0) ** 2


def grad_f(x):
    return 2.0 * (x - 3.0)


def run_optimizer(args, optimizer, init_x, name="optimizer"):
    x = np.array([init_x], dtype=np.float64)

    print(f"\n=== {name} ===")

    for step in range(args.steps):
        loss = f(x)
        grad = grad_f(x)

        print(
            "Step {:3d}  x {:8.6f}  loss {:8.6f}  grad {:8.6f}".format(
                step,
                float(x[0]),
                float(loss[0]),
                float(grad[0])
            )
        )

        optimizer.update(x, grad)


def main():
    parser = argparse.ArgumentParser(
        description="Compare SGD and STAGE-SGD on quadratic optimization"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="number of optimization steps (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="learning rate (default: 0.1)",
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
        "--init-x",
        type=float,
        default=10.0,
        help="initial value of x (default: 10.0)",
    )

    args = parser.parse_args()

    sgd = SGD(lr=args.lr)
    stage_sgd = stage.optim.numpy.StageSGD(
        lr=args.lr,
        tau=args.tau,
        eps=args.eps,
    )

    run_optimizer(args, sgd, args.init_x, "SGD")
    run_optimizer(args, stage_sgd, args.init_x, "STAGE-SGD")


if __name__ == "__main__":
    main()
