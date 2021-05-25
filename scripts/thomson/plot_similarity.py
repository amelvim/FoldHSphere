
import matplotlib.pyplot as plt
import seaborn as sns


def plot_matsim_hist(cos_mat, cosines):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot cosine similarity matrix NxN
    img = ax1.matshow(
        cos_mat, vmin=-1, vmax=1, interpolation="nearest", cmap="YlGnBu"
    )
    ax1.set_xlabel("Cosine Similarity Scores")
    plt.colorbar(img, ax=ax1, orientation="horizontal", pad=0.08)

    # Plot cosine similarity histogram
    sns.histplot(cosines, ax=ax2, bins=100, binrange=(-1, 1),
                 stat="probability", alpha=0.4, edgecolor="gray",
                 linewidth=0.2, kde=True, line_kws={"linewidth": 2})
    ax2.set_xlabel("Cosine Similarity Scores")
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(0, 0.6)
    ax2.grid(linestyle="--", alpha=0.5)

    # Plot cummulative cosine distance histogram
    sns.histplot(1-cosines, ax=ax3, bins=100, cumulative=True,
                 stat="probability", alpha=0.4, edgecolor="gray",
                 line_kws={"linewidth": 2})
    ax3.set_xlabel("Cosine Distances Cummulative")
    ax3.set_xlim(0, 2)
    ax3.set_ylim(0, 1)
    ax3.grid(linestyle="--", alpha=0.5)

    return fig
