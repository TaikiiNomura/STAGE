# examples\quadratic\quadratic_sgd.py

import argparse
import numpy as np
import matplotlib.pyplot as plt

import stage


def f(x):
    return (x - 3.0) ** 2


def grad_f(x):
    return 2.0 * (x - 3.0)


def train_model(args, x, optimizer):
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

        if args.verbose and step % args.log_interval == 0:
            print(
                "Step {} x {:.6f} loss {:.6f} grad {:.6f}".format(
                    step,
                    float(x[0]),
                    float(loss[0]),
                    float(grad[0]),
                )
            )

    return xs, losses, grads


def main():
    parser = argparse.ArgumentParser(description="STAGE NumPy quadratic optimization")

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
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="logging interval",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show optimization logs",
    )

    args = parser.parse_args()

    x = np.array([args.init_x], dtype=np.float64)

    optimizer = stage.optim.numpy.StageSGD(
        lr=args.lr,
        tau=args.tau,
        eps=args.eps,
    )

    xs, losses, grads = train_model(args, x, optimizer)

    # plot 1: function curve and optimization trajectory
    x_curve = np.linspace(-2.0, 10.0, 400)
    y_curve = (x_curve - 3.0) ** 2

    plt.figure(figsize=(8, 5))
    plt.plot(x_curve, y_curve, label="f(x) = (x - 3)^2")

    traj_x = np.array(xs)
    traj_y = (traj_x - 3.0) ** 2
    plt.plot(traj_x, traj_y, marker="o", label="optimization trajectory")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Quadratic Optimization Trajectory")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # plot 2: x over steps
    plt.figure(figsize=(8, 5))
    plt.plot(xs, label="x")
    plt.axhline(y=3.0, linestyle="--", label="optimal x = 3")
    plt.xlabel("Step")
    plt.ylabel("x")
    plt.title("Parameter Trajectory")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # plot 3: loss over steps
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="loss")
    plt.xlabel("Step")
    plt.ylabel("f(x)")
    plt.title("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # plot 4: gradient over steps
    plt.figure(figsize=(8, 5))
    plt.plot(grads, label="gradient")
    plt.axhline(y=0.0, linestyle="--", label="optimal grad = 0")
    plt.xlabel("Step")
    plt.ylabel("grad f(x)")
    plt.title("Gradient")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()