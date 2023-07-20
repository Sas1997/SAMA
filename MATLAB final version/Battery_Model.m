
function [Pch_max,Pdch_max]=Battery_Model(Cn_B,Nbat,Eb,alfa_battery,c,k,Imax,Vnom,ef_bat)

dt=1;       % the length of the time step [h]
Q1=c*Eb ;   % the available energy [kWh] in the storage at the beginning of the time step
Q=Eb;       % the total amount of energy [kWh] in the storage at the beginning of the time step
Qmax=Cn_B;  % is the total capacity of the storage bank [kWh]
Nbatt=Nbat;   % the number of batteries in the storage bank 

Pch_max1=-(-k*c*Qmax+k*Q1*exp(-k*dt)+Q*k*c*(1-exp(-k*dt)))/(1-exp(-k*dt)+c*(k*dt-1+exp(-k*dt)));
Pch_max2=(1-exp(-alfa_battery*dt))*(Qmax-Q)/dt;
Pch_max3=Nbatt*Imax*Vnom/1000;

Pdch_max=(k*Q1*exp(-k*dt)+Q*k*c*(1-exp(-k*dt)))*sqrt(ef_bat)/(1-exp(-k*dt)+c*(k*dt-1+exp(-k*dt)));

Pch_max=min([Pch_max1 Pch_max2 Pch_max3])/sqrt(ef_bat);

end