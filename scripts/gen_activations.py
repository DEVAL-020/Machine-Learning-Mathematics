import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

x = np.linspace(-6, 6, 400)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)
leaky_relu = np.where(x > 0, x, 0.1 * x)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

configs = [
    (sigmoid, "Sigmoid", r"$\sigma(x)=\dfrac{1}{1+e^{-x}}$", "#2A5CAA"),
    (tanh, "Tanh", r"$\tanh(x)$", "#C0392B"),
    (relu, "ReLU", r"$\max(0,x)$", "#27AE60"),
    (leaky_relu, "Leaky ReLU", r"$\max(0.1x,\,x)$", "#8E44AD"),
]

for ax, (y, title, formula, color) in zip(axes, configs):
    ax.plot(x, y, color=color, linewidth=2.5)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title(f"{title}\n{formula}", fontsize=13)
    ax.set_xlabel("$x$")
    ax.set_ylim(-1.5, 6) if title != "Sigmoid" and title != "Tanh" else ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig("figures/activation_functions.png", dpi=200, bbox_inches="tight")
print("saved activation_functions.png")
