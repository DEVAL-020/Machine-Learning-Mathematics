import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})
np.random.seed(3)
x = np.linspace(0, 10, 40)
y = 2.2 * x + 3 + np.random.normal(0, 2.2, size=x.shape)

A = np.vstack([x, np.ones_like(x)]).T
theta, *_ = np.linalg.lstsq(A, y, rcond=None)
y_hat = theta[0] * x + theta[1]

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x, y, color="#2A5CAA", s=35, alpha=0.8, label="Training data")
ax.plot(x, y_hat, color="#C0392B", linewidth=2.5,
        label=f"$\\hat y = {theta[0]:.2f}x + {theta[1]:.2f}$")
for xi, yi, yhi in zip(x, y, y_hat):
    ax.plot([xi, xi], [yi, yhi], color="gray", linewidth=0.6, alpha=0.6)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("Least-Squares Linear Regression (residuals shown in gray)")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/linear_regression_fit.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved linear_regression_fit.png")

np.random.seed(1)
n = 60
class0 = np.random.normal(loc=[-1.5, -1.5], scale=1.0, size=(n, 2))
class1 = np.random.normal(loc=[1.5, 1.5], scale=1.0, size=(n, 2))
X = np.vstack([class0, class1])
Y = np.hstack([np.zeros(n), np.ones(n)])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w = np.array([0.1, 0.1])
b = 0.0
lr = 0.1
Xb = np.hstack([X, np.ones((2 * n, 1))])
wb = np.array([0.1, 0.1, 0.0])
for _ in range(2000):
    z = Xb @ wb
    p = sigmoid(z)
    grad = Xb.T @ (p - Y) / len(Y)
    wb -= lr * grad

xx, yy = np.meshgrid(np.linspace(-5, 5, 300), np.linspace(-5, 5, 300))
grid = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)]
probs = sigmoid(grid @ wb).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 6))
cf = ax.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.55)
ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2)
ax.scatter(class0[:, 0], class0[:, 1], color="#2A5CAA", edgecolor="white", s=45, label="Class 0")
ax.scatter(class1[:, 0], class1[:, 1], color="#C0392B", edgecolor="white", s=45, label="Class 1")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Logistic Regression Decision Boundary")
ax.legend(fontsize=11, loc="upper left")
cbar = fig.colorbar(cf, ax=ax)
cbar.set_label("$P(y=1\\mid x)$")
plt.tight_layout()
plt.savefig("figures/logistic_decision_boundary.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved logistic_decision_boundary.png")
