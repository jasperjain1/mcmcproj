import numpy as np
import matplotlib.pyplot as plt

#parspace=np.array([H0R,AsR,omch2R,tauR,ombh2R,w,wa,ns])

data = np.loadtxt('omg.csv', delimiter=',', dtype=float)
step=np.linspace(1,len(data[:,0]),len(data[:,0]))
Ass=data[:,6]
ws=data[:,5]
print(np.shape(step))
print(np.shape(Ass))
plt.scatter(Ass,ws,c=step/20)
plt.show()