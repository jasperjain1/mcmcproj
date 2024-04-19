import camb
import numpy as np
import matplotlib.pyplot as plt
import scipy
data=np.loadtxt('spectrum.txt')
ls=data[:,0]
Ds=data[:,1]
sigmas=(data[:,2]+data[:,3])/2
print(camb.dark_energy.DarkEnergyEqnOfState.set_params.__doc__)

pars = camb.set_params(H0=65.04687, ombh2=.02221, omch2=.118464, tau=.065311,  
                       As=2.13245e-9, ns=.96654, halofit_version='mead', lmax=3000, dark_energy_model='DarkEnergyPPF')
pars.DarkEnergy = camb.DarkEnergyPPF(w=-.850362, wa=-.55497)


results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL=np.resize(np.delete(powers['total'][:,0],[0,1]),(len(ls)))

pars = camb.set_params(H0=67.4, ombh2=.0224, omch2=.120, tau=.054,  
                       As=2.0905e-9, ns=.9626, halofit_version='mead', lmax=3000, dark_energy_model='DarkEnergyPPF')
pars.DarkEnergy = camb.DarkEnergyPPF(w=-1, wa=0)


results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL2=np.resize(np.delete(powers['total'][:,0],[0,1]),(len(ls)))

print(np.sum(((Ds-totCL)/sigmas)**2))
print(np.sum(((Ds-totCL2)/sigmas)**2))


#plt.plot(ls,scipy.signal.savgol_filter(np.abs(totCL-Ds),100,2), label = "ours")

#plt.plot(ls,scipy.signal.savgol_filter(np.abs(totCL2-Ds),100,2))
# plt.plot(ls,totCL-totCL2)

# plt.legend()
# plt.show()