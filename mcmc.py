import numpy as np
import camb
import matplotlib.pyplot as plt
import scipy

#Loading the Planck 2020 TT angular power spectrum
data=data=np.loadtxt('spectrum.txt')
ls=data[:,0]
Ds=data[:,1]
lmax=800
ls=np.resize(ls, (lmax))
Ds=np.resize(Ds,(lmax))
#For l<30, the error bars given by Planck are asymmetric. I have no will to deal with that. Thus, 
sigmas=(data[:,2]+data[:,3])/2

sigmas=np.resize(sigmas,(lmax))
n=lmax
#Parameters and ranges 
#H0: 62-72
H0R= np.array([60,75])
#As: 1.9-2.1e-9
AsR=np.array([1.95e-9,2.25e-9])
#omch2 (dark matter density, matter density fixed at ombh2=.022): .1-.14 
omch2R=np.array([.1,.16])
#tau: .045-.075
tauR=np.array([.04,.08])
#omk:-.1 - .1
omkR=np.array([-.1,.1])
w=np.array([-1.3,-.7])
wa=np.array([-1,1])
ns=np.array([.7,1.3])
parspace=np.array([H0R,AsR,omch2R,tauR,omkR,w,wa,ns])

#how long to run the mcmc
numSteps=1500

def chi2(params):
    """
    Given an array "params" of params, returns the likelihood that point in parameter space 
    """
    H0=params[0]
    As=params[1]
    omch2=params[2]
    tau=params[3]
    omk=params[4]
    w=params[5]
    wa=params[6]
    ns=params[7]

    
    pars = camb.set_params(H0=H0, ombh2=.022, omch2=omch2, omk=omk, mnu=0.06, tau=tau,  
                       As=As, ns=ns, halofit_version='mead', lmax=lmax)
    pars.DarkEnergy=camb.DarkEnergyPPF(w=w,wa=wa)

    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL=np.resize(np.delete(powers['total'][:,0],[0,1]),(lmax))
    chi2=np.sum(((Ds-totCL)/sigmas)**2)
    return chi2
    
    #for large degrees of freedom, the chi squared distribution becomes approximately normal with mean df and variance 2df, so
def lhood(chi2):  
    return 1-scipy.stats.norm.cdf((chi2-n)/np.sqrt(2*n))

#to be used in the proposal function: checks if the proposed paramters are inside the paramter space
def inParspace(nparams, parspace):
    for i in range(len(parspace)):
        if (nparams[i]<parspace[i][0] or nparams[i]>parspace[i][1]):
            return False
    return True

#proposal function
def proposal(params, parspace):
    while True:
        seed=np.random.rand((len(parspace)))-.5
        step=(parspace[:,1]-parspace[:,0])*seed/17
        if inParspace(params+step,parspace):
            return params+step

#test to accept or reject point: uses current likelihood instead of params so that this likelihood doesn't need to be recalculated each time
#if True, accept the new parameters. If False, reject.
def accept(chi2Current, nparams):
    chi2New=chi2(nparams)

    #This is a mess, because often it will lhood=0 for both. Thus, as will be described in the paper, we use L'Hopital's rule.
    
    #Case where new params are better.
    if chi2Current>chi2New:
        return True
    
    lCurrent=lhood(chi2Current)
    lNew=lhood(chi2New)
    u=np.random.rand()

    print("Current likelihood:" + str(lCurrent) + "  prop lhood:" + str(lNew) + "  prop chi2:" + str(chi2New))
    #Case where old params are better, but both likelihoods are 0. See the paper.
    if (lNew == 0 and lCurrent == 0):
        alpha=np.exp(-(1/2)*((chi2New-n)**2-(chi2Current-n)**2)/2*n)
        if u<alpha:
            return True
        return False
    
    #Case where old params are better, and they don't both give lhood=0 
    alpha=lNew/lCurrent
    if u<alpha: 
        return True
    return False

startSeed=np.random.rand(len(parspace))
params=parspace[:,0]+(parspace[:,1]-parspace[:,0])*startSeed
print("Starting paramters: " + str(params) )

distribution=[]

iterations=0
while iterations < numSteps:
    print("Accepted")
    accepted=False
    chi2Current= chi2(params)
    while accepted==False:
        distribution.append(params)
        print("Current parameters: " + str(params))
        print("Current Chi Squared" + str(chi2Current))

        nparams = proposal(params,parspace)
        accepted = accept(chi2Current, nparams)
        iterations+=1
        print(iterations)
        if iterations % 30 == 0:
            np.savetxt('data5.csv',np.array(distribution),delimiter=',')
    params=nparams

print(distribution)




"""

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