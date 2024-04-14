import numpy as np
import camb
import matplotlib.pyplot as plt
import cv2


ws = np.linspace(63,70, 10)
data=data=np.loadtxt('spectrum.txt')
ls=data[:,0]
Ds=data[:,1]
sigmas=(data[:,2]+data[:,3])/2

a=np.zeros(len(ws))
j=0
for w in ws: 
    pars = camb.set_params(H0=w, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,  
                       As=2.1e-9, ns=0.965, halofit_version='mead', lmax=2501)
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL=powers['total']
    #lsim = np.arange(totCL.shape[0])
    #plt.plot(lsim, totCL[:,0])
    chi2=0
    for i in range(1,2000):
        chi2+=((Ds[i]-totCL[i,0])/sigmas[i])**2
    a[j]= chi2
    j+=1



plt.plot(ws,a)
plt.show()

#for i in range(len(Dl)):
 #   Cl[i]=1/(ls[i]*(ls[i]+1))*Dl[i]
#plt.errorbar(ls,Ds,yerr=sigmas)
#plt.show()

#Computing likelihoods:


print("done")
