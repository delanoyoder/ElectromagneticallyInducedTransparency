import numpy as np
from constants import *
from cmath import phase
from scipy.integrate import complex_ode
import matplotlib.pyplot as plt

class TLS:

    def __init__(self, t, c_1, c_2, c_3):
        self.t = t
        self.c_1 = c_1
        self.c_2 = c_2
        self.c_3 = c_3

class EIT:

    def __init__(self):
        self.DE = complex_ode(self.three_level_system)

    def three_level_system(self, t, c, A=[50,50], probe_offset=5):
        
        c_1, c_2, c_3 = c
        A_1, A_2 = A
        
        O_c = A_1 * (np.tanh(t-1) + 1)
        O_p = A_2 * (np.tanh(t-1 - probe_offset) + 1)
        
        dcdt = [(i / 2 * (O_p * c_3))]
        dcdt.append((i / 2 * (O_c * c_3)))
        dcdt.append((i / 2 * (O_p * c_1 + O_c * c_2)))
    
        return dcdt
    
    def simulate(self, populations_0=[1,0,0], t_0=0, t_f=10, dt=0.25):

        self.DE.set_initial_value(populations_0, t_0)

        self.c_1 = []
        self.c_2 = []
        self.c_3 = []
        self.t = []
        while self.DE.successful() and self.DE.t < t_f:
            self.DE.integrate(self.DE.t+dt)
            self.c_1.append(abs(self.DE.y[0]))
            self.c_2.append(abs(self.DE.y[1]))
            self.c_3.append(abs(self.DE.y[2]))
            self.t.append(self.DE.t)

    def simulate_phase_transfer(self):
        pass

    def plot_dark_state(self):
        plt.figure(figsize=(7,5))    
        plt.plot(self.t,self.c_1)
        plt.plot(self.t,self.c_2,'--')
        plt.plot(self.t,self.c_3,'.')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.legend([r'$c_{1}$',r'$c_{2}$',r'$c_{3}$'])
        plt.show()

    def plot_phase_transfer(self):
        pass