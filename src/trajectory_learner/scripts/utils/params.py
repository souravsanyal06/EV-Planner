#!/usr/bin/env python3
import sys
import numpy as np

R = 0.3
#Supply Voltage
v = 15 #V
rpm = 7994
#Motor Friction Torque
T_f = v/rpm

C_T = 0.0048

C_Q = C_T * np.sqrt(C_T/2)
rho = 1.225
r = 0.1
A = np.pi*r**2
#Aerodynamic drag coefficient
k_Tau = C_Q*rho*A*r**3
#Viscous damping coefficint
D_f = 2e-04
#Voltage constant of motor
K_E = rpm/v
K_T = K_E
J_m = 4.9e-06
n_B = 3
m_B = 0.001

eps = 0.023
J_L = 0.25*n_B*m_B*(r-eps)**2
J = J_m + J_L


c1 = R*T_f**2/K_T**2
c2 = T_f/K_T*(2*R*D_f/K_T + K_E)
c3 = (D_f/K_T)*(R*D_f/K_T + K_E) + (2*R*T_f*k_Tau)/K_T**2

c4 = (k_Tau/K_T)*(2*R*D_f/K_T + K_E)

c5 = R*k_Tau**2/K_T**2
c6 = R*J**2/K_T**2
# w0 = float(sys.argv[1])
# print(c1)
# print(c2*w0)
# print(c3*w0**2)
# print(K_E)
# print(c4)
# print(c4*w0**3)
# print(c5*w0**4)
# print(c6)
# i=0
# w = np.array([w0, w0, w0, w0])
# print(c1 + c2 * w[i] + c3 * w[i]**2 + c4 * w[i]**3 + c5 * w[i]**4)