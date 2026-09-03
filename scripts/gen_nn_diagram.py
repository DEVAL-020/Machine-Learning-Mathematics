import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams.update({"font.size": 12})

layer_sizes = [3, 4, 4, 2]
layer_names = ["Input\nLayer", "Hidden\nLayer 1", "Hidden\nLayer 2", "Output\nLayer"]
v_spacing = 1.2
h_spacing = 2.8

fig, ax = plt.subplots(figsize=(11, 6))

positions = []
for li, n in enumerate(layer_sizes):
    layer_y = [ (i - (n - 1) / 2) * v_spacing for i in range(n) ]
    positions.append(layer_y)

colors = ["#2A5CAA", "#27AE60", "#27AE60", "#C0392B"]

for li in range(len(layer_sizes) - 1):
    x0, x1 = li * h_spacing, (li + 1) * h_spacing
    for y0 in positions[li]:
        for y1 in positions[li + 1]:
            ax.plot([x0, x1], [y0, y1], color="gray", linewidth=0.6, alpha=0.5, zorder=1)

for li, (n, ys) in enumerate(zip(layer_sizes, positions)):
    x = li * h_spacing
    for y in ys:
        circ = Circle((x, y), 0.28, color=colors[li], ec="black", linewidth=1.2, zorder=3)
        ax.add_patch(circ)
    ax.text(x, max(ys) + v_spacing * 0.9, layer_names[li], ha="center", fontsize=12, fontweight="bold")

ax.annotate(
    r"$a^{(2)}_1 = \sigma\!\left(\sum_i w^{(1)}_{1i}\,a^{(1)}_i + b^{(1)}_1\right)$",
    xy=(h_spacing, positions[1][0]), xytext=(h_spacing * 0.9, -3.6),
    fontsize=12, ha="center",
    arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2),
)

ax.set_xlim(-1, (len(layer_sizes) - 1) * h_spacing + 1)
ax.set_ylim(-4.2, 3.2)
ax.axis("off")
ax.set_title("Feedforward Neural Network Architecture", fontsize=15, pad=15)

plt.tight_layout()
plt.savefig("figures/neural_network_diagram.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved neural_network_diagram.png")
