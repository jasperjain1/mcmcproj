import numpy as np
import matplotlib.pyplot as plt
import scipy

#parspace=np.array([H0R,AsR,omch2R,tauR,ombh2R,w,wa,ns])

data1 = np.loadtxt('data10-15.csv', delimiter=',', dtype=float)
data2 = np.loadtxt('data13-13.csv', delimiter=',', dtype=float)
data3 = np.loadtxt('data14-5.csv', delimiter=',', dtype=float)

H0R= np.array([50,90])
#As: 1.9-2.1e-9
AsR=np.array([1.7e-9,2.3e-9])
#omch2 (dark matter density, matter density fixed at ombh2=.022): .1-.14 
omch2R=np.array([.06,.18])
#tau: .045-.075
tauR=np.array([.0442,.08])
#omk:-.1 - .1
ombh2R=np.array([.013,.035])
w=np.array([-1.5,-.5])
#new param: wa=w-3wa
wa=np.array([-1,1])
ns=np.array([.7,1])
parspace=np.array([H0R,AsR,omch2R,tauR,ombh2R,w,wa,ns])

figure, axis = plt.subplots(2,4) 
parlabels= ["$H_0$", "$A_s$", "$\Omega_ch^2$", "$\tau$", "$\Omega_bh^2$", "$w_0$", "$w_a$", "$n_s$"]
for i in range(len(data1[0,:])):
    d1=data1[1000:2500,i]
    d2=data2[1000:2500,i]
    d3=data3[1000:2500,i]
    comb=np.concatenate((d1,d2,d3))
    axis[i//4,i%4].hist(comb,bins=30)
    axis[i//4,i%4].set_xlabel(parlabels[i])
    axis[i//4,i%4].set_ylabel("Frequency")
       #(mu, sigma) = scipy.stats.norm.fit(comb)
    #axis[i//4,i%4].plt(e^)
    plt.setp(axis[i//4,i%4].get_yticklabels(), visible=False)

plt.show()

""""
step=np.linspace(1,len(data[:,0]),len(data[:,0]))
Ass=data[:,6]
ws=data[:,5]
print(np.shape(step))
print(np.shape(Ass))
plt.scatter(Ass,ws,c=step/20)
plt.show()"""