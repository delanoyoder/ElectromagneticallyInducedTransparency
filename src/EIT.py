from os import environ
from cmath import phase
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode
from numpy import pi, array, arange, mean, sin, cos, tanh


class ElectromagneticallyInducedTransparency:
    PLOT_DIR = f"{environ['VIRTUAL_ENV'][:-5]}/plots"

    def __init__(self, probe, coupling):
        self.probe = probe
        self.coupling = coupling

    def differential_equation(self, t, c):
        """Calculates the differential equation for electromagnetically induced transparency.

        Args:
            t (float): Time.
            c (list): List of complex numbers representing the state of the system.

        Returns:
            list: List of complex numbers representing the time differential state of the system.
        """
        i = complex(0, 1)

        Ωp = self.rabi_frequency(t, self.probe)
        Ωc = self.rabi_frequency(t, self.coupling)

        dcdt1 = (i / 2) * Ωp * c[2]
        dcdt2 = (i / 2) * Ωc * c[2]
        dcdt3 = (i / 2) * (Ωp * c[0] + Ωc * c[1])

        return [dcdt1, dcdt2, dcdt3]

    #########################
    # Static Methods ########
    #########################

    @staticmethod
    def rabi_frequency(t, laser):
        """Returns the Rabi frequency of a laser at a particular time.

        Args:
            t (float): Time.
            laser (dict): Dictionary of laser parameters.
                {time offset, amplitude}

        Returns:
            float: Rabi frequency.
        """
        return (laser["A"] * (tanh(t - laser["t"] - pi) + 1))

    @staticmethod
    def normalize(x):
        """Normalizes a list of numbers.

        Args:
            x (list): List of numbers.

        Returns:
            list: Normalized list of numbers.
        """        
        return (x - min(x)) / (max(x) - min(x))


class DarkState(ElectromagneticallyInducedTransparency):
    def __init__(self, probe={"t": pi, "A": 50}, coupling={"t": 0, "A": 50}):
        super().__init__(probe, coupling)
        self.t = []
        self.dcdt = {"1": [], "2": [], "3": []}
        self.Ω = {"p": [], "c": []}

    def simulate(self, c0=[1, 0, 0], t0=0, tf=10, dt=0.25):
        """Simulates the dark state of a system.

        Args:
            c0 (list, optional): Initial state of the system. Defaults to [1, 0, 0].
            t0 (int, optional): Initial time of the simulation. Defaults to 0.
            tf (int, optional): Final time of the simulation. Defaults to 10.
            dt (float, optional): Time step. Defaults to 0.25.
        """
        ode = complex_ode(self.differential_equation)
        ode.set_initial_value(c0, t0)
        while ode.successful() and ode.t < tf:
            ode.integrate(ode.t + dt)
            self.t.append(ode.t)
            self.dcdt["1"].append(abs(ode.y[0]))
            self.dcdt["2"].append(abs(ode.y[1]))
            self.dcdt["3"].append(abs(ode.y[2]))

        self.Ω["c"] = self.rabi_frequency(array(self.t), self.coupling)
        self.Ω["p"] = self.rabi_frequency(array(self.t), self.probe)

    def plot(self, save_directory=None):
        """Plots the results of a dark state simulation.

        Args:
            results (dict): Dictionary of results to be plotted.
            save_directory (str, optional): Directory to save the plot to. Defaults to None.
        """
        fig, ax1 = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(7)

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Population")
        ax1.plot(self.t, self.dcdt["1"], "k")
        ax1.plot(self.t, self.dcdt["2"], "b--")
        ax1.plot(self.t, self.dcdt["3"], "g.")
        ax1.tick_params(axis="y")
        ax1.legend([r"$c_{1}$", r"$c_{2}$", r"$c_{3}$"], loc=2)

        ax2 = ax1.twinx()

        ax2.set_ylabel(r"$Ω$ (arb. units)")
        ax2.plot(self.t, self.Ω["c"], "r:", alpha=0.5)
        ax2.plot(self.t, self.Ω["p"], "y-.", alpha=0.5)
        ax2.tick_params(axis="y")
        ax2.legend([r"$Ω_{C}$", r"$Ω_{P}$"], loc=1)

        fig.tight_layout()

        if save_directory is not None:
            plt.savefig(f"{save_directory}/dark_state_simulation.png")

        plt.show()


class PhaseTransfer(ElectromagneticallyInducedTransparency):
    def __init__(self, probe={"t": 5, "A": 5}, coupling={"t": 0, "A": 25}):
        super().__init__(probe, coupling)
        self.x = arange(pi / 2, 3 * pi / 2, 0.001)
        self.amplitudes = coupling["A"] * sin(self.x)
        self.φ = []

    def simulate(self, **kwargs):
        """Simulates the phase transfer of a system."""        
        for amplitude in self.amplitudes:
            self.coupling["A"] = amplitude
            self.φ.append(self.compute_phase(**kwargs))

    def compute_phase(self, c0=[1, 0, 0], t0=0, tf=10, dt=0.01):
        """Returns the average phase of a system.

        Args:
            c0 (list, optional): Initial state of the system. Defaults to [1, 0, 0].
            t0 (int, optional): Initial time of the simulation. Defaults to 0.
            tf (int, optional): Final time of the simulation. Defaults to 10.
            dt (float, optional): Time step. Defaults to 0.25.

        Returns:
            float: Average phase.
        """        
        ode = complex_ode(self.differential_equation)
        ode.set_initial_value(c0, t0)
        φ = []
        while ode.successful() and ode.t < tf:
            ode.integrate(ode.t + dt)
            φ.append(phase(ode.y[0]))
        return mean(φ)

    def plot(self, save_directory=None):
        """Plots the results of a phase transfer simulation.

        Args:
            save_directory (str, optional): Directory to save the plot to. Defaults to None.
        """        
        self.x = self.x / pi - 1

        fig, ax1 = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(7)
        color = "tab:red"
        ax1.set_xlabel("x (microns)")
        ax1.set_ylabel(r"$Ω_{C}$ (arb. units)", color=color)
        ax1.plot(self.x, self.normalize(self.amplitudes), color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel(r"Phase of $c_{1}$ (arb. units)", color=color)
        ax2.plot(self.x, self.normalize(self.φ), color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()

        if save_directory is not None:
            plt.savefig(f"{save_directory}/spatial_offset_vs_phase_transfer.png")

        plt.show()


if __name__ == "__main__":
    DSS = DarkState()
    DSS.simulate()
    DSS.plot(save_directory=DSS.PLOT_DIR)

    PTS = PhaseTransfer()
    PTS.simulate()
    PTS.plot(save_directory=PTS.PLOT_DIR)
