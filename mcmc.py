import numpy 
import camb
params = camb.model.CAMBparams(ombh2=.28, omch2=0, w=-4)
print(camb.get_age(params))

print(params)