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

def chi2(params):
    """
    Given an array "params" of params, returns the likelihood that point in parameter space 
    """
    H0=params[0]
    As=params[1]
    omch2=params[2]
    tau=params[3]
    ombh2=params[4]
    w=params[5]
    wa=params[6]
    ns=params[7]

    
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau,  
                       As=As, ns=ns, halofit_version='mead', lmax=lmax)
    pars.DarkEnergy=camb.DarkEnergyPPF(w=w,wa=wa)

    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL=np.resize(np.delete(powers['total'][:,0],[0,1]),(lmax))
    chi2=np.sum(((Ds-totCL)/sigmas)**2)
    return chi2

def lhood(chi2):  
    return 1-scipy.stats.norm.cdf((chi2-n)/np.sqrt(2*n))


"parspace=np.array([H0R,AsR,omch2R,tauR,ombh2R,w,wa,ns])"

params=np.array([67.5,2.14e-9,.122,.06,.02230,-1,0,.965])
print(lhood(chi2(params)))

print(lhood(880))