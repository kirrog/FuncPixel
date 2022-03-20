import sys
import timeit
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf

from data_work.data_loader import pict_size, load_grad_local_minims, load_local_vals

data = load_local_vals()
limit = 0.0005
coef = 1000000
print("Data loaded")
print("Max: " + str(data.max()))
print("Min: " + str(data.min()))
exit()
res = np.zeros((data.shape[1], int(limit * coef)), dtype=int)
num = np.zeros(data.shape[1])
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        grad = abs(data[i, j, 0]) + abs(data[i, j, 1]) / 2
        if grad < limit:
            res[j, int(grad * coef)] += 1
        else:
            num[j] += 1
print(num)
x_number = [x / coef for x in range(int(limit * coef))]
for j in range(data.shape[1]):
    plt.plot(x_number, res[j])
    plt.xlabel("Positions")
    plt.ylabel("Points number")
    plt.savefig(
        "classes_minimums_vals/class{c:01d}/ limit: {p:05f} coef: {coef:05d}.jpg".format(c=j, p=limit, coef=coef))
    plt.figure().clear()
f = open("results/point_values_analysis.md", "a")
f.write("| {limit:03f} | {coef:010d} ".format(limit=limit, coef=coef))
for i in num:
    f.write("| {n:05d} ".format(n=int(i)))
f.write("|\n")
