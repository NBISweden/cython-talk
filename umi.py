import sys
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.spatial import distance
from scipy.cluster import hierarchy


def main():
    with open(sys.argv[1]) as f:
        umis = [line.strip() for line in f]
    figure = plot_clustermap(umis[:2000])
    figure.savefig("plot.png")


def plot_clustermap(sequences):
    matrix = distances(sequences, distfunc=hamming_distance_py)
    linkage = hierarchy.linkage(distance.squareform(matrix), method='average')
    fig = sns.clustermap(
        matrix,
        row_linkage=linkage,
        col_linkage=linkage,
        linewidths=None,
        linecolor='none',
        figsize=(500/25.4, 500/25.4),
        xticklabels=False,
        yticklabels=False
    )
    return fig


def hamming_distance_py(s, t):
    if len(s) != len(t):
        raise IndexError("Sequences must have the same length")
    dist = 0
    for c, d in zip(s, t):
        if c != d:
            dist += 1
    return dist


def distances(sequences, distfunc):
    """Compute all pairwise Hamming distances"""
    m = np.zeros((len(sequences), len(sequences)), dtype=float)
    for i, s in enumerate(sequences):
        for j, t in enumerate(sequences):
            if i < j:
                m[j, i] = m[i, j] = distfunc(s, t)
    return pd.DataFrame(m)


if __name__ == "__main__":
    main()
