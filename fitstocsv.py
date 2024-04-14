
import camb
import numpy as np
import scipy


a=np.array([0,1,2])
b=np.array([2,3,4])
c=np.array([4,4,4])

print(np.sum(((a-b)/c)**2))

"""
pars = camb.set_params(H0=67, ombh2=.022, omch2=1.2, mnu=0.06, omk=0, tau=.066,  
                       As=2.1e-9, ns=0.965, halofit_version='mead', lmax=3000)
results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL=np.resize(np.delete(powers['total'][:,0],[0,1]),(2507))
print(len(totCL))"""