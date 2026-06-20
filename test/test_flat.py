import numpy as np

x = np.arange(10).reshape(-1, 1)
x = np.stack([x,x], axis=0)


print("x.shape", x.shape)
print(x[0, :, 0])
print(x[1, :, 0])
print("====="*5)

x_flat = x.reshape(-1, x.shape[-1])

for i in range(x_flat.shape[1]):
    print(x_flat[:, i])
