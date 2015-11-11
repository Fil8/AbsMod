# modello
This is the 1dimensional model for a disk with I=90 and PA=0
we generate the analytical emission line of Stewart 2014:


f(v)=2F_{tot}/ (PI DV) * 1/rho(u)

rho^2(u)= 1-u^2 for |u|<1 
rho^2(u)= 0     else

DV = range between maximum and minimum gas velocities

u = v-v_0/(DV/2)

!!!! when the binning is equal to 1, otherwise the normalization factor changes.

How to: 
run the program:
 - python model_1d.py

 -  in the rootdirectory there must be a parameter file: par.txt
