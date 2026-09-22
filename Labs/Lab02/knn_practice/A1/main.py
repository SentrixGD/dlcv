import random
import time

import dlcv2026
import matplotlib.pyplot as plt
import torch
import torchvision
from knn import compute_distances_no_loops, compute_distances_one_loop, compute_distances_two_loops

plt.rcParams["figure.figsize"] = (10.0, 8.0)
plt.rcParams["font.size"] = 16

x_train: torch.Tensor
y_train: torch.Tensor
x_test: torch.Tensor
y_test: torch.Tensor

x_train, y_train, x_test, y_test = dlcv2026.data.cifar10()


print(
    "Training set:",
)
print("  data shape:", x_train.shape)
print("  labels shape: ", y_train.shape)
print("Test set:")
print("  data shape: ", x_test.shape)
print("  labels shape", y_test.shape)


classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
samples_per_class = 12
samples = []
for y, cls in enumerate(classes):
    plt.text(-4, 34 * y + 18, cls, ha="right")
    (idxs,) = (y_train == y).nonzero(as_tuple=True)
    for i in range(samples_per_class):
        idx = int(idxs[random.randrange(idxs.shape[0])].item())
        samples.append(x_train[idx])
img = torchvision.utils.make_grid(samples, nrow=samples_per_class)
plt.imshow(dlcv2026.tensor_to_image(img))
plt.axis("off")
plt.show()

help(dlcv2026.data.cifar10)

num_train = 500
num_test = 250

x_train, y_train, x_test, y_test = dlcv2026.data.cifar10(num_train, num_test)

print(
    "Training set:",
)
print("  data shape:", x_train.shape)
print("  labels shape: ", y_train.shape)
print("Test set:")
print("  data shape: ", x_test.shape)
print("  labels shape", y_test.shape)


torch.manual_seed(0)
num_train = 500
num_test = 250
x_train, y_train, x_test, y_test = dlcv2026.data.cifar10(num_train, num_test)

dists = compute_distances_two_loops(x_train, x_test)
print("dists has shape: ", dists.shape)

x_train_small = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float64)
x_test_small = torch.tensor([[0.0, 4.0], [3.0, 0.0]], dtype=torch.float64)
expected = torch.tensor([[16.0, 9.0], [9.0, 16.0]], dtype=torch.float64)

actual = compute_distances_two_loops(x_train_small, x_test_small)
assert actual.shape == (2, 2), f"Expected shape (2, 2), got {tuple(actual.shape)}"
assert torch.allclose(actual, expected), f"Expected {expected}, got {actual}"
print("Two-loop distance test passed.")


torch.manual_seed(0)
x_train_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)
x_test_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)

dists_one = compute_distances_one_loop(x_train_rand, x_test_rand)
dists_two = compute_distances_two_loops(x_train_rand, x_test_rand)
difference = (dists_one - dists_two).pow(2).sum().sqrt().item()
print("Difference: ", difference)
if difference < 1e-4:
    print("Good! The distance matrices match")
else:
    print("Uh-oh! The distance matrices are different")

torch.manual_seed(0)
x_train_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)
x_test_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)

dists_two = compute_distances_two_loops(x_train_rand, x_test_rand)
dists_none = compute_distances_no_loops(x_train_rand, x_test_rand)
difference = (dists_two - dists_none).pow(2).sum().sqrt().item()
print("Difference: ", difference)
if difference < 1e-10:
    print("Good! The distance matrices match")
else:
    print("Uh-oh! The distance matrices are different")


def timeit(f, *args):
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


torch.manual_seed(0)
x_train_rand = torch.randn(500, 3, 32, 32)
x_test_rand = torch.randn(500, 3, 32, 32)

two_loop_time = timeit(compute_distances_two_loops, x_train_rand, x_test_rand)
print(f"Two loop version took {two_loop_time:.2f} seconds")

one_loop_time = timeit(compute_distances_one_loop, x_train_rand, x_test_rand)
speedup = two_loop_time / one_loop_time
print(f"One loop version took {one_loop_time:.2f} seconds ({speedup:.1f}X speedup)")

no_loop_time = timeit(compute_distances_no_loops, x_train_rand, x_test_rand)
speedup = two_loop_time / no_loop_time
print(f"No loop version took {no_loop_time:.2f} seconds ({speedup:.1f}X speedup)")
