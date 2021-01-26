import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.sin(2 * x)
y4 = np.sin(4 * x)

plt.polar(x, y1, label='sin(x)')
plt.polar(x, y2, label='sin(x)')
plt.polar(x, y4, label='sin(x)')

plt.title('Polar sinX')

plt.xticks(np.linspace(0, 2 * np.pi, 5), ['0', 'π/2', 'π', '3π/2', ''])
plt.legend()
plt.show()