import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})

fig, ax = plt.subplots(figsize=(6, 6))
a = np.array([4, 1])
b = np.array([2, 3])
proj_scalar = np.dot(a, b) / np.dot(b, b)
proj = proj_scalar * b

origin = np.array([0, 0])
ax.quiver(*origin, *a, angles="xy", scale_units="xy", scale=1, color="#2A5CAA", width=0.012, label=r"$\mathbf{a}$")
ax.quiver(*origin, *b, angles="xy", scale_units="xy", scale=1, color="#27AE60", width=0.012, label=r"$\mathbf{b}$")
ax.quiver(*origin, *proj, angles="xy", scale_units="xy", scale=1, color="#C0392B", width=0.012,
          label=r"$\mathrm{proj}_{\mathbf{b}}\,\mathbf{a}$")
ax.plot([a[0], proj[0]], [a[1], proj[1]], "--", color="gray", linewidth=1.2)

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 4)
ax.axhline(0, color="black", linewidth=0.6)
ax.axvline(0, color="black", linewidth=0.6)
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.legend(fontsize=12, loc="upper left")
ax.set_title("Vector Projection of $\\mathbf{a}$ onto $\\mathbf{b}$")
plt.tight_layout()
plt.savefig("figures/vector_projection.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved vector_projection.png")

A = np.array([[2, 0.5], [0.5, 1.5]])
eigvals, eigvecs = np.linalg.eig(A)

theta = np.linspace(0, 2 * np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])
ellipse = A @ circle

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.plot(circle[0], circle[1], color="lightgray", linewidth=1.5, label="Unit circle")
ax.plot(ellipse[0], ellipse[1], color="#2A5CAA", linewidth=2, label="Transformed by $A$")

colors = ["#C0392B", "#27AE60"]
for i in range(2):
    v = eigvecs[:, i]
    lam = eigvals[i]
    ax.quiver(0, 0, *v, angles="xy", scale_units="xy", scale=1, color=colors[i],
              width=0.012, label=f"Eigenvector $v_{i+1}$ ($\\lambda_{i+1}={lam:.2f}$)")
    ax.quiver(0, 0, *(lam * v), angles="xy", scale_units="xy", scale=1, color=colors[i],
              width=0.006, alpha=0.5)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.axhline(0, color="black", linewidth=0.6)
ax.axvline(0, color="black", linewidth=0.6)
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.legend(fontsize=10, loc="upper left")
ax.set_title("Eigenvectors of $A$ Stay on Their Own Span\n(A = [[2, 0.5], [0.5, 1.5]])", fontsize=13)
plt.tight_layout()
plt.savefig("figures/eigenvectors.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved eigenvectors.png")
