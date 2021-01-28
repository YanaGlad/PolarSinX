import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(-1, 1, 50)
y = x

z = x.reshape(-1, 1) * y

plt.contour(x,y,z)
plt.show()
