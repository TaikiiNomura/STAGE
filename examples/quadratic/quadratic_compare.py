# examples\quadratic\quadratic_compare.py

import argparse
import numpy as np
import matplotlib.pyplot as plt

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


def run_optimizer(args, optimizer, init_x):
    x = np.array([init_x], dtype=np.float64)

    xs = []
    losses = []
    grads = []

    for step in range(args.steps):
        loss = f(x)
        grad = grad_f(x)

        xs.append(float(x[0]))
        losses.append(float(loss[0]))
        grads.append(float(grad[0]))

        optimizer.update(x, grad)

    return xs, losses, grads


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

    sgd_xs, sgd_losses, sgd_grads = run_optimizer(args, sgd, args.init_x)
    stage_xs, stage_losses, stage_grads = run_optimizer(args, stage_sgd, args.init_x)

    # plot 1: function curve and trajectories
    x_curve = np.linspace(-2.0, 10.0, 400)
    y_curve = (x_curve - 3.0) ** 2

    plt.figure(figsize=(8, 5))
    plt.plot(x_curve, y_curve, label="f(x) = (x - 3)^2")

    sgd_traj_x = np.array(sgd_xs)
    sgd_traj_y = (sgd_traj_x - 3.0) ** 2
    plt.plot(sgd_traj_x, sgd_traj_y, marker="o", markersize=4, label="SGD")

    stage_traj_x = np.array(stage_xs)
    stage_traj_y = (stage_traj_x - 3.0) ** 2
    plt.plot(stage_traj_x, stage_traj_y, marker="x", markersize=5, label="STAGE-SGD")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Quadratic Optimization Trajectory")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # plot 2: x over steps
    plt.figure(figsize=(8, 5))
    plt.plot(sgd_xs, label="SGD")
    plt.plot(stage_xs, label="STAGE-SGD")
    plt.axhline(y=3.0, linestyle="--", label="optimal x = 3")
    plt.xlabel("Step")
    plt.ylabel("x")
    plt.title("Parameter Trajectory")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # plot 3: loss over steps
    plt.figure(figsize=(8, 5))
    plt.plot(sgd_losses, label="SGD")
    plt.plot(stage_losses, label="STAGE-SGD")
    plt.xlabel("Step")
    plt.ylabel("f(x)")
    plt.title("Loss Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # plot 4: gradient over steps
    plt.figure(figsize=(8, 5))
    plt.plot(sgd_grads, label="SGD")
    plt.plot(stage_grads, label="STAGE-SGD")
    plt.axhline(y=0.0, linestyle="--", label="optimal grad = 0")
    plt.xlabel("Step")
    plt.ylabel("grad f(x)")
    plt.title("Gradient Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()