import sys
import pickle
import numpy as np
from scipy.spatial import distance


def calculate_similarity(feat1, feat2):
    ''' Calculate cosine similarity between two feature vectors '''
    return 1 - distance.cosine(feat1, feat2)


def main(names_file, embedding_file, score_file):
    # read names
    names = np.loadtxt(names_file, dtype="str")

    # parse pairs
    pairs = [(i, j) for i in names for j in names if i != j]

    # load features dictionary
    with open(embedding_file, "rb") as h:
        featdict = pickle.load(h)

    # calculate similarity for each pair
    with open(score_file, "w") as fout:
        for i, p in enumerate(pairs):
            sim = calculate_similarity(featdict[p[0]], featdict[p[1]])
            print(p[0], p[1], sim, file=fout)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: %s <names_file> <embedding_file> <score_file>"
                 % sys.argv[0])
    print(sys.argv[1:])
    names_file, embedding_file, score_file = sys.argv[1:]

    main(names_file, embedding_file, score_file)
