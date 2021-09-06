import numpy as np
from constants import *
from cmath import phase
from scipy.integrate import complex_ode
import matplotlib.pyplot as plt


class TLS:
    def __init__(self):

        self.t = []
        self.c_1 = []
        self.c_2 = []
        self.c_3 = []
        self.O_c = []
        self.O_p = []

    def plot_DSS(self):

        fig, ax1 = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(7)

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Population")
        ax1.plot(self.t, self.c_1, "k")
        ax1.plot(self.t, self.c_2, "b--")
        ax1.plot(self.t, self.c_3, "g.")
        ax1.tick_params(axis="y")
        ax1.legend([r"$c_{1}$", r"$c_{2}$", r"$c_{3}$"], loc=2)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel(r"$Ω$ (arb. units)")  # we already handled the x-label with ax1
        ax2.plot(self.t, self.O_c, "r:", alpha=0.5)
        ax2.plot(self.t, self.O_p, "y-.", alpha=0.5)
        ax2.tick_params(axis="y")
        ax2.legend([r"$Ω_{C}$", r"$Ω_{P}$"], loc=1)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()


class EIT:
    def __init__(self, t_0, t_f, dt, c_0, t_c, A_c, t_p, A_p, TLS):

        self.t_0 = t_0
        self.t_f = t_f
        self.dt = dt
        self.c_0 = c_0
        self.t_c = t_c
        self.t_p = t_p
        self.A_c = A_c
        self.A_p = A_p
        self.TLS = TLS

        self.ode = complex_ode(self.diff_eq)

    def diff_eq(self, t, c):

        c_1 = (i / 2) * (self.probe_laser(t) * c[2])
        c_2 = (i / 2) * (self.coupling_laser(t) * c[2])
        c_3 = (i / 2) * (self.probe_laser(t) * c[0] + self.coupling_laser(t) * c[1])

        return [c_1, c_2, c_3]

    def coupling_laser(self, t):
        return self.A_c * (np.tanh(t - self.t_c - np.pi) + 1)

    def probe_laser(self, t):
        return self.A_p * (np.tanh(t - self.t_p - np.pi) + 1)

    def DSS(self):

        self.ode.set_initial_value(self.c_0, self.t_0)

        while self.ode.successful() and self.ode.t < self.t_f:

            self.ode.integrate(self.ode.t + self.dt)

            self.TLS.t.append(self.ode.t)
            self.TLS.c_1.append(abs(self.ode.y[0]))
            self.TLS.c_2.append(abs(self.ode.y[1]))
            self.TLS.c_3.append(abs(self.ode.y[2]))

        self.TLS.O_c = self.coupling_laser(np.array(self.TLS.t))
        self.TLS.O_p = self.probe_laser(np.array(self.TLS.t))

    def PTS(self, x_0, x_f, A_0, A_f):

        x = np.arange(x_0, x_f, (x_f - x_0) / 100)


"""
from EIT import *
b = TLS()
a = EIT(0, 10, 0.25, [1,0,0], 0, 50, np.pi, 50, b)
a.DSS()
b.plot_DSS()
"""
