import matplotlib.pyplot as plt
import numpy as np

y = [8.1022,7.0301,4.5729,3.9827,4.8418,4.6934,3.7779,4.0987,3.4502,4.2199,2.7747,2.8484,2.8216,3.4804,2.9456]
x = np.linspace(1,len(y),len(y),endpoint = True)

plt.plot(x,y,'r-',linewidth=1)
# plt.xscale('log')
plt.title("Loss for Different Iters")
plt.xlabel("Iters")
plt.ylabel("Loss")
plt.show()
