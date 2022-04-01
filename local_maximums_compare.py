import sys

import faiss
import numpy as np

path_cutted = "data/cutted_models_maximums/class_{c:02d}.npy"
path_orig = "data/unique/{c:02d}.npy"
pict_size = 28
dim = pict_size * pict_size
topn = 100


def distilate_to_unique_maximums(orig, cutted, cl):
    index = faiss.IndexFlatL2(dim)
    index.add(orig)
    before = orig.shape[0]
    after = cutted.shape[0]
    merge = 0
    D, I = index.search(cutted, k=topn)
    for i in range(len(D)):
        if D[i][0] < dim * 0.1:
            merge += 1
    print("Class: {cl:02d} before: {bef:06d} after: {aft:06d} merge: {mer:06d}".format(cl=cl, bef=before, aft=after,
                                                                                       mer=merge))


for i in range(10):
    orig = np.load(path_orig.format(c=i))
    cutted = np.load(path_cutted.format(c=i))
    distilate_to_unique_maximums(orig, cutted, i)
