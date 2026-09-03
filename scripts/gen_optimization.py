import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({"font.size": 12})

def f(x, y):
    return x**2 + 3 * y**2

def grad_f(x, y):
    return np.array([2 * x, 6 * y])

pos = np.array([4.0, 2.5])
lr = 0.1
path = [pos.copy()]
for _ in range(25):
    pos = pos - lr * grad_f(*pos)
    path.append(pos.copy())
path = np.array(path)

x = np.linspace(-5, 5, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig, ax = plt.subplots(figsize=(7, 6))
cs = ax.contour(X, Y, Z, levels=20, cmap="viridis")
ax.plot(path[:, 0], path[:, 1], "o-", color="#C0392B", markersize=4,
        linewidth=1.5, label="Gradient descent steps")
ax.plot(path[0, 0], path[0, 1], "s", color="black", markersize=8, label="Start")
ax.plot(0, 0, "*", color="gold", markersize=16, markeredgecolor="black", label="Minimum")
ax.set_xlabel("$\\theta_1$")
ax.set_ylabel("$\\theta_2$")
ax.set_title("Gradient Descent on $f(\\theta_1,\\theta_2) = \\theta_1^2 + 3\\theta_2^2$")
ax.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig("figures/gradient_descent_contour.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved gradient_descent_contour.png")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, linewidth=0, antialiased=True)
ax.plot(path[:, 0], path[:, 1], f(path[:, 0], path[:, 1]) + 0.5,
        color="#C0392B", linewidth=2.5, marker="o", markersize=3, label="GD trajectory")
ax.set_xlabel("$\\theta_1$")
ax.set_ylabel("$\\theta_2$")
ax.set_zlabel("$J(\\theta)$")
ax.set_title("Loss Surface and Descent Trajectory")
ax.view_init(elev=35, azim=-60)
plt.tight_layout()
plt.savefig("figures/loss_surface_3d.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved loss_surface_3d.png")
