import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})

x = np.linspace(-10, 10, 500)

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

fig, ax = plt.subplots(figsize=(7, 5))
params = [(0, 1, "#2A5CAA"), (0, 2, "#27AE60"), (2, 1, "#C0392B")]
for mu, sigma, color in params:
    ax.plot(x, gaussian(x, mu, sigma), color=color, linewidth=2.3,
            label=f"$\\mu={mu},\\ \\sigma={sigma}$")
    ax.fill_between(x, gaussian(x, mu, sigma), alpha=0.08, color=color)

ax.set_xlabel("$x$")
ax.set_ylabel("$p(x)$")
ax.set_title(r"Gaussian PDF: $p(x)=\dfrac{1}{\sigma\sqrt{2\pi}}\,e^{-\frac{(x-\mu)^2}{2\sigma^2}}$")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/gaussian_distributions.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved gaussian_distributions.png")

np.random.seed(7)
true_mu, true_sigma = 3.0, 1.5
samples = np.random.normal(true_mu, true_sigma, 40)

mu_grid = np.linspace(0, 6, 300)
sigma_fixed = true_sigma
log_likelihood = np.array([
    np.sum(np.log(gaussian(samples, mu, sigma_fixed))) for mu in mu_grid
])
best_mu = mu_grid[np.argmax(log_likelihood)]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

axes[0].hist(samples, bins=12, density=True, color="#2A5CAA", alpha=0.5, edgecolor="white", label="Observed data")
xx = np.linspace(-2, 8, 300)
axes[0].plot(xx, gaussian(xx, best_mu, sigma_fixed), color="#C0392B", linewidth=2.3,
             label=f"Fitted $\\mathcal{{N}}(\\hat\\mu={best_mu:.2f}, \\sigma)$")
axes[0].set_title("Data and MLE-Fitted Distribution")
axes[0].set_xlabel("$x$")
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

axes[1].plot(mu_grid, log_likelihood, color="#27AE60", linewidth=2.3)
axes[1].axvline(best_mu, color="#C0392B", linestyle="--", linewidth=1.5,
                 label=f"$\\hat\\mu_{{MLE}}={best_mu:.2f}$")
axes[1].set_title("Log-Likelihood as a Function of $\\mu$")
axes[1].set_xlabel("$\\mu$")
axes[1].set_ylabel(r"$\log \mathcal{L}(\mu)$")
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/mle_illustration.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved mle_illustration.png")
