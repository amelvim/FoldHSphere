import sys
import numpy as np

score_file = sys.argv[1]
cvdir = "data/lindahl/cv2_fold_level/"

# Load scores
scores = np.loadtxt(score_file, dtype=str)
scores_dict = {item[0] + " " + item[1]: item[2] for item in scores}
del scores

# Compute top 1 prediction accuracy
acc = []
for pairs_file in ["LE_1_2.pairs_labels", "LE_2_1.pairs_labels"]:
    # Load pair lists with binary labels
    labels_set = np.loadtxt(cvdir + pairs_file, dtype=str)
    # Filter scores by pair lists
    scores_set = np.array([[i, j, scores_dict[i + " " + j]] \
                           for i, j in labels_set[:,:2]])
    # Get number of correct predictions
    names = np.unique(labels_set[:,0])
    correct = 0
    for n in names:
        pos = np.argwhere(labels_set[:,0] == n).flatten()
        preds_n = scores_set[pos]
        labels_n = labels_set[pos]
        pos_top1 = np.argsort(preds_n[:,2])[-1]
        pair_pred = preds_n[pos_top1]
        pair_label = labels_n[pos_top1]
        if pair_label[-1] == "1":
            correct += 1
    acc.append(correct / names.size)

print("Accuracy per set:\n\tLE_1: %.4f\n\tLE_2: %.4f" % (acc[0], acc[1]))
print("Average accuracy: %.4f" % (sum(acc) / 2))
