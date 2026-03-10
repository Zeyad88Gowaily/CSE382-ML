import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from matplotlib.lines import Line2D

# Mean Vectors
mu1 = np.array([-1, 3])
mu2 = np.array([0, 6])
mu3 = np.array([-2, 4])

# Covariance Matrices
sigma1 = np.array([[1, -0.5],
                   [-0.5, 2]])

sigma2 = np.array([[2, -2],
                   [-2, 7]])

sigma3 = np.array([[1, 1.5],
                   [1.5, 3]])


GIVEN_PRIORS = [1/4, 1/4, 1/2] # GIVEN in the assignment instructions


# Define grid
x, y = np.meshgrid(np.linspace(-10, 5, 200), np.linspace(0, 10, 300))
pos = np.dstack((x, y))

# Distributions
pdf1 = mvn(mu1, sigma1).pdf(pos)
pdf2 = mvn(mu2, sigma2).pdf(pos)
pdf3 = mvn(mu3, sigma3).pdf(pos)    

def plot_decision(ax, priors, title):
    post1 = pdf1 * priors[0]
    post2 = pdf2 * priors[1]
    post3 = pdf3 * priors[2]

    decision = np.argmax(np.array([post1, post2, post3]), axis=0)

    step = 3  
    for cls, color, marker in zip([0, 1, 2], ['red', 'green', 'blue'], ['o', '+', '*']):
        mask = decision[::step, ::step] == cls
        ax.scatter(x[::step, ::step][mask], y[::step, ::step][mask],
                   c=color, marker=marker, s=10, linewidths=0.5)

    # draw a boundary line where each pair of posteriors are equal
    ax.contour(x, y, post1 - post2, levels=[0], colors='black',linewidths=1.5, linestyles='--')
    ax.contour(x, y, post1 - post3, levels=[0], colors='purple',linewidths=1.5, linestyles='--')
    ax.contour(x, y, post2 - post3, levels=[0], colors='yellow',linewidths=1.5, linestyles='--')

    for mu, color in zip([mu1, mu2, mu3], ['red', 'green', 'blue']):
        ax.plot(*mu, 'X', color=color, markersize=12, markeredgecolor='black', zorder=5)

    class_handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='red',markersize=8, label='Class 1'),
        Line2D([0],[0], marker='+', color='w', markerfacecolor='green',markersize=8, label='Class 2'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor='blue',markersize=8, label='Class 3'),
    ]
    boundary_handles = [
        Line2D([0],[0], color='black',linestyle='--', linewidth=1.5, label='Boundary 1|2'),
        Line2D([0],[0], color='purple',linestyle='--', linewidth=1.5, label='Boundary 1|3'),
        Line2D([0],[0], color='yellow',linestyle='--', linewidth=1.5, label='Boundary 2|3'),
    ]
    ax.legend(handles=class_handles + boundary_handles, fontsize=7, loc='upper right')

    prior_str = f"P(c1)={priors[0]:.2f}, P(c2)={priors[1]:.2f}, P(c3)={priors[2]:.2f}"
    ax.set_title(f"{title}\n{prior_str}", fontsize=10)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(-10, 5)
    ax.set_ylim(0, 10)


# Part (a): Likelihood Contours
fig, ax = plt.subplots(figsize=(8, 6))

for pdf, mu, color, label in zip(
        [pdf1, pdf2, pdf3],
        [mu1, mu2, mu3],
        ['red', 'green', 'blue'],
        ['Class 1', 'Class 2', 'Class 3']):

    cs = ax.contour(x, y, pdf, levels=6, colors=color, linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.4f')
    ax.plot(*mu, 'X', color=color, markersize=12, markeredgecolor='black', zorder=5, label=f'{label} mean')

ax.set_title("(a) Likelihood Contours")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_xlim(-10, 5)
ax.set_ylim(0, 10)
ax.legend(fontsize=9)
ax.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig('part_a_likelihood_contours.png', dpi=150)
plt.show()


# Part (b): Decision Map using assignment given priors
fig, ax = plt.subplots(figsize=(8, 6))
plot_decision(ax, GIVEN_PRIORS, "(b) Decision Map")
ax.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig('part_b_decision_map.png', dpi=150)
plt.show()


# Part (c): 4 scenarios to show how priors shift the decision regions
scenarios = [
    (GIVEN_PRIORS,       "Original (1/4, 1/4, 1/2)"),
    ([1/3, 1/3, 1/3],    "Equal Priors"),
    ([0.6, 0.3, 0.1],    "Class 1 Favored"),
    ([0.1, 0.1, 0.8],    "Class 3 Heavily Favored"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("(c) Effect of Changing Prior Probabilities", fontsize=13, fontweight='bold')

for ax, (priors, title) in zip(axes.flat, scenarios):
    plot_decision(ax, priors, title)
    ax.grid(True, linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig('part_c_prior_effect.png', dpi=150)
plt.show()


# Addtional Analysis: Compute posteriors for the original priors to use in parts (d) and (e).
def compute_posteriors(priors):
    """Return normalised posterior arrays (H, W) for each class."""
    p1 = pdf1 * priors[0]
    p2 = pdf2 * priors[1]
    p3 = pdf3 * priors[2]
    total = p1 + p2 + p3 + 1e-300          #1e-300 to avoid division by zero
    return p1 / total, p2 / total, p3 / total


# ── Part (d): Posterior Probability Heatmaps ──
# Shows HOW CONFIDENT the classifier is at every grid point for each class.
post1_n, post2_n, post3_n = compute_posteriors(GIVEN_PRIORS)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("(d) Posterior Probability Heatmaps  [P(class | x)]",
             fontsize=13, fontweight='bold')

for ax, post, mu, title, cmap in zip(
        axes,
        [post1_n, post2_n, post3_n],
        [mu1, mu2, mu3],
        ['Class 1  P(c₁|x)', 'Class 2  P(c₂|x)', 'Class 3  P(c₃|x)'],
        ['Reds', 'Greens', 'Blues']):

    im = ax.imshow(post, extent=[-10, 5, 0, 10], origin='lower',
                   aspect='auto', cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.plot(*mu, 'X', color='black', markersize=12, markeredgecolor='white',
            zorder=5, label='Mean')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('part_d_posterior_heatmaps.png', dpi=150)
plt.show()


# ── Part (e): Bayes Error Map ──
# At each point the Bayes error = 1 - max posterior  (minimum achievable error).
bayes_error = 1.0 - np.maximum.reduce([post1_n, post2_n, post3_n])

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(bayes_error, extent=[-10, 5, 0, 10], origin='lower',
               aspect='auto', cmap='hot_r', vmin=0, vmax=1)
fig.colorbar(im, ax=ax, label='Bayes Error  1 − max P(cₖ|x)')

# overlay decision boundaries
p1 = pdf1 * GIVEN_PRIORS[0];  p2 = pdf2 * GIVEN_PRIORS[1];  p3 = pdf3 * GIVEN_PRIORS[2]
ax.contour(x, y, p1 - p2, levels=[0], colors='cyan',    linewidths=1.5, linestyles='--')
ax.contour(x, y, p1 - p3, levels=[0], colors='lime',    linewidths=1.5, linestyles='--')
ax.contour(x, y, p2 - p3, levels=[0], colors='magenta', linewidths=1.5, linestyles='--')

for mu, color in zip([mu1, mu2, mu3], ['red', 'green', 'blue']):
    ax.plot(*mu, 'X', color=color, markersize=12, markeredgecolor='black', zorder=5)

mean_error = bayes_error.mean()
ax.set_title(f"(e) Bayes Error Map\nMean error over grid = {mean_error:.4f}", fontsize=11)
ax.set_xlabel("x1");  ax.set_ylabel("x2")
ax.set_xlim(-10, 5);  ax.set_ylim(0, 10)
ax.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()
plt.savefig('part_e_bayes_error_map.png', dpi=150)
plt.show()
print(f"\n[Part e]  Mean Bayes error over the grid: {mean_error:.4f}  ({mean_error*100:.2f}%)")


# ── Part (f): Confusion Matrix ──
# Sample N points from each class, run the Bayes classifier, build confusion matrix.
np.random.seed(42)
N_SAMPLES = 2000          # samples per class

dist1 = mvn(mu1, sigma1)
dist2 = mvn(mu2, sigma2)
dist3 = mvn(mu3, sigma3)

samples = {
    0: dist1.rvs(N_SAMPLES),
    1: dist2.rvs(N_SAMPLES),
    2: dist3.rvs(N_SAMPLES),
}

confusion = np.zeros((3, 3), dtype=int)

for true_cls, pts in samples.items():
    l1 = dist1.pdf(pts) * GIVEN_PRIORS[0]
    l2 = dist2.pdf(pts) * GIVEN_PRIORS[1]
    l3 = dist3.pdf(pts) * GIVEN_PRIORS[2]
    predicted = np.argmax(np.stack([l1, l2, l3], axis=1), axis=1)
    for pred_cls in range(3):
        confusion[true_cls, pred_cls] = np.sum(predicted == pred_cls)

accuracy= np.trace(confusion) / confusion.sum()
per_cls_acc= confusion.diagonal() / confusion.sum(axis=1)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(confusion, cmap='Blues')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

class_labels = ['Class 1', 'Class 2', 'Class 3']
ax.set_xticks([0, 1, 2]);  ax.set_xticklabels(class_labels, fontsize=11)
ax.set_yticks([0, 1, 2]);  ax.set_yticklabels(class_labels, fontsize=11)
ax.set_xlabel("Predicted Class", fontsize=12)
ax.set_ylabel("True Class",      fontsize=12)
ax.set_title(f"(f) Confusion Matrix (N={N_SAMPLES} per class)\n"
             f"Overall Accuracy = {accuracy*100:.2f}%", fontsize=11)

thresh = confusion.max() / 2.0
for i in range(3):
    for j in range(3):
        pct = confusion[i, j] / confusion[i].sum() * 100
        ax.text(j, i, f"{confusion[i,j]}\n({pct:.1f}%)",
                ha='center', va='center', fontsize=10,
                color='white' if confusion[i, j] > thresh else 'black')

plt.tight_layout()
plt.savefig('part_f_confusion_matrix.png', dpi=150)
plt.show()