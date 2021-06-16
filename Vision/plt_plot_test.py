import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
x = np.arange(3)
y = [1,2,3]
plt.plot(x, y)
plt.xlabel("frame")
plt.ylabel("familiarity")
plt.show()
