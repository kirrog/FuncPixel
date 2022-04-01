import sys

import faiss
import numpy as np
from pathlib import Path
from data_work.data_loader import load_x_model_vectors, load_cutted_model_vectors

pict_size = 28
dim = pict_size * pict_size
topn = 100
lr = 0.1
min_dist = lr * dim


def find_distances(vectors):
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    D, I = index.search(vectors, k=topn)
    return D, I


def distilate_to_unique_maximums(full_vectors):
    vectors = full_vectors
    unique = []
    deleted = 0
    iteration = 0
    while True:
        D, I = find_distances(vectors)
        indexes_to_delete = []
        sys.stdout.write("Founded")
        for i in range(len(D)):
            for j in range(1, len(D[i])):
                if D[i][j] < min_dist and I[i][j] > i:
                    indexes_to_delete.append(I[i][j])
                elif j == 1:
                    unique.append(vectors[i])
                    indexes_to_delete.append(i)
        indexes_to_delete = list(set(indexes_to_delete))
        size = vectors.shape[0]
        new_size = size - len(indexes_to_delete)
        deleted += len(indexes_to_delete)
        sys.stdout.write(
            "\rIteration: num: {num:06d} unique: {unique_num:06d} deleted: {dele:06d} ".format(num=iteration,
                                                                                               unique_num=len(
                                                                                                   unique),
                                                                                               dele=deleted))
        if new_size == 0:
            print()
            break
        new_vectors_list = list(range(size))
        for i in indexes_to_delete:
            new_vectors_list.remove(i)
        new_vectors = np.zeros((new_size, dim))
        assert len(new_vectors_list) == new_size
        for i in range(len(new_vectors_list)):
            new_vectors[i] = vectors[new_vectors_list[i]]

        vectors = np.array(new_vectors, dtype=np.float32)
        print(vectors.shape)
        iteration += 1

    return np.vstack(unique)


def count_base_model_maximums():
    x_model_vectors_raw = load_x_model_vectors()

    for i in range(1, 9):
        print("Class {i:02d}".format(i=i))
        x_model_vectors = np.array(np.reshape(x_model_vectors_raw[:, i],
                                              (x_model_vectors_raw.shape[0],
                                               pict_size * pict_size)), dtype=np.float32)
        res = distilate_to_unique_maximums(x_model_vectors)
        np.save("data/unique/{i:02d}.npy".format(i=i), res)

def count_cutted_models_maximums():
    for i in range(10):
        print("Class {i:02d}".format(i=i))
        x_model_vectors_raw = load_cutted_model_vectors(i)
        x_model_vectors = np.array(np.reshape(x_model_vectors_raw,
                                              (x_model_vectors_raw.shape[0],
                                               pict_size * pict_size)), dtype=np.float32)
        print("Loaded")
        res = distilate_to_unique_maximums(x_model_vectors)
        print("Calculated")
        path_class = "data/cutted_models_maximums/class_{class_num:02d}.npy".format(class_num=i)
        Path("data/cutted_models_maximums/").mkdir(parents=True, exist_ok=True)
        np.save(path_class, res)

count_cutted_models_maximums()
