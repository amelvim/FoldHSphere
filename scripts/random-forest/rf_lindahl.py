import pickle
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def append_scores(sim_file, scores_files):
    with open(sim_file, "rb") as f:
        featsdict = pickle.load(f)
    for scfile in scores_files:
        with open(scfile) as f:
            lines = f.read().splitlines()
        for l in lines:
            x = l.split(" ")
            name = x[0] + " " + x[1]
            score = float(x[2])
            featsdict[name] = np.append(featsdict[name], score)
    return featsdict


def main(sim_file, scores_files, train_file, test_file, save_file, njobs):

    print("[*] Loading all data (Lindahl dataset)")
    scores_files = list(map(str, scores_files.split("+")))
    featsdict = append_scores(sim_file, scores_files)

    print("[*] Separating data in train-test")
    with open(train_file, "r") as f:
        train_pairs = f.read().splitlines()
    with open(test_file, "r") as f:
        test_pairs = f.read().splitlines()

    feats_train = np.array([featsdict[item] for item in train_pairs])
    X_train = feats_train[:,1:]
    y_train = feats_train[:,0]

    feats_test = np.array([featsdict[item] for item in test_pairs])
    X_test = feats_test[:,1:]
    y_test = feats_test[:,0]

    clf = RandomForestClassifier(
        n_estimators=500, random_state=0, n_jobs=int(njobs)
    )
    print("[*] Training...")
    clf.fit(X_train, y_train)
    print("[*] Testing...")
    y_prob = clf.predict_proba(X_test)

    print("[*] Saving probability results")
    ids = np.array(test_pairs)
    output = np.vstack((ids, y_prob[:,1])).T
    np.savetxt(save_file, output, delimiter=" ", fmt="%s")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit("Usage: %s <sim_file> <scores_files> <train_file> "
                 "<test_file> <save_file> <njobs>" % sys.argv[0])
    print(sys.argv[1:])
    sim_file, scores_files, train_file, test_file, save_file, njobs = sys.argv[1:]

    main(sim_file, scores_files, train_file, test_file, save_file, njobs)
