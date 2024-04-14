import numpy as np
import camb
import matplotlib.pyplot as plt
import scipy

#Loading the Planck 2020 TT angular power spectrum
data=data=np.loadtxt('spectrum.txt')
ls=data[:,0]
Ds=data[:,1]
#For l<30, the error bars given by Planck are asymmetric. I have no will to deal with that. Thus, 
sigmas=(data[:,2]+data[:,3])/2
n=len(ls)
#Parameters and ranges 
#H0: 62-72
H0R= [62,72]
#As: 1.9-2.1e-9
AsR=[1.9e-9,2.1e-9]
#omch2 (dark matter density, matter density fixed at ombh2=.022): .1-.14 
omch2R=[.1,.14]
#tau: .045-.075
tauR=[.045,.075]
#omk:-.1 - .1
omkR=[-.1,.1]
parspace=[H0R,AsR,omch2R,tauR,omkR]

#how long to run the mcmc
steps=10

def lhood(params):
    """
    Given an array "params" of params, returns the likelihood that point in parameter space 
    """
    H0=params[0]
    As=params[1]
    omch2=params[2]
    tau=params[3]
    omk=params[4]
    pars = camb.set_params(H0=H0, ombh2=.022, omch2=omch2, omk=omk, mnu=0.06, tau=tau,  
                       As=As, ns=0.965, halofit_version='mead', lmax=2501)
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL=np.resize(np.delete(powers['total'][:,0],[0,1]),(2507))
    chi2=np.sum(((Ds-totCL)/sigmas)**2)
    #for large degrees of freedom, the chi squared distribution becomes approximately normal with mean df and variance 2df, so
    return scipy.stats.norm.cdf((chi2-n)/np.sqrt(2*n))

#to be used in the proposal function: checks if the proposed paramters are inside the paramter space
def inParspace(nparams, parspace):
    for i in len(parspace):
        if (nparams[i]<parspace[i][0] or nparams[i]>parspace[i][1]):
            return False
    return True

#proposal function
def proposal(params, parspace):
    while True:
        step=np.random.rand((len(parspace)))
        if inParspace(params+step,parspace):
            return params+step

#test to accept or reject point: uses current likelihood instead of params so that this likelihood doesn't need to be recalculated each time
#if True, accept the new parameters. If False, reject.
def accept(lCurrent, nparams)
    lNew=lhood(nparams)
    if lNew>lCurrent:
        return True
    if lNew<lCurrent:
        alpha=lNew/lCurrent
        u=np.random.rand()
        if u<alpha: 
            return True
    return False

startpoint=np.rand()
for step in range(steps):
    lCurrent=lhood(params)




print(1-lhood([66,2.1e-9,.122,.066,0]))
"""ws=np.linspace(60,70,10)



a=np.zeros(len(ws))
j=0
    print(pars)
    
    
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
"""