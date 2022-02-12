import numpy as np


def create_generator_points(dim, side_size):
    num_of_squares = int((side_size * side_size) / (dim * dim))
    num_of_points = pow(2, num_of_squares)

    def generator_points():
        for i in range(num_of_points):
            result = np.zeros((side_size, side_size))
            for j in range(num_of_squares):
                if (i // pow(2, j)) % 2 == 1:
                    x = (j % side_size) % (side_size // dim) * dim
                    y = (j // (side_size // dim)) * dim
                    result[x:x + dim, y:y + dim] = 1
            yield result, i

    return generator_points(), num_of_points
