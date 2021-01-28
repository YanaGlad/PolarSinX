import numpy as np
import matplotlib.pyplot as plt


def F(x):
    return x ** 4 + 5 * x ** 3 - 10 * x


x = np.linspace(-5, 2, 100)
y = list(map(F, x))

plt.figure(figsize=(10, 7))
plt.plot(x, y)
plt.ylabel("Y")
plt.xlabel("X")
plt.scatter([-3.5518, -0.9439, 0.7457], [F(-3.5518), F(-0.9439), F(0.7457)], linewidths=5)
plt.show()
