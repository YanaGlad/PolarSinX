import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.sin(2 * x)
y4 = np.sin(4 * x)

plt.figure(figsize=(14, 8))

plt.plot(x, y1, label='sinx')
plt.plot(x, y2, label='sin2x')
plt.plot(x, y4, label='sin4x')

plt.title('Basic sinX')
plt.xlabel('x')
plt.ylabel('sinx')
plt.xticks(np.linspace(0, 2 * np.pi, 5), ['0', 'π/2', 'π', '3π/2', '2π'])
plt.legend()
plt.show()