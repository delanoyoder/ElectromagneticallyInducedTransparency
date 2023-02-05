from os import environ
from cmath import phase
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode
from numpy import pi, array, arange, mean, sin, tanh


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

    def simulate_transfer(self, c0=[1, 0, 0], t0=0, tf=10, dt=0.01):
        """Returns the phases/populations of a system over the time.

        Args:
            c0 (list, optional): Initial state of the system. Defaults to [1, 0, 0].
            t0 (int, optional): Initial time of the simulation. Defaults to 0.
            tf (int, optional): Final time of the simulation. Defaults to 10.
            dt (float, optional): Time step. Defaults to 0.01.

        Returns:
            list, list, list: The phase, imaginary phase, and population transfers of the system.
        """        
        ode = complex_ode(self.differential_equation)
        ode.set_initial_value(c0, t0)

        transfer = {
            "time": [],
            "phase": {"x": [], "y": [], "z": []},
            "population": {"x": [], "y": [], "z": []},
        }

        while ode.successful() and ode.t < tf:
            ode.integrate(ode.t + dt)
            transfer["time"].append(ode.t)
            for i, axis in enumerate(["x", "y", "z"]):
                transfer["phase"][axis].append(phase(ode.y[i]))
                transfer["population"][axis].append(abs(ode.y[i]))

        return transfer

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
        return (laser["A"] * (tanh(t - laser["t"]) + 1))

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
    def __init__(self, probe={"t": 2 * pi, "A": 50}, coupling={"t": pi, "A": 50}):
        super().__init__(probe, coupling)
        self.t = []
        self.dcdt = {"x": [], "y": [], "z": []}
        self.Ω = {"p": [], "c": []}

    def simulate(self, **kwargs):
        """Simulates the dark state of a system."""
        transfer = self.simulate_transfer(dt=0.25, **kwargs)
        self.t = transfer["time"]
        self.dcdt = transfer["population"]
        self.Ω["c"] = self.rabi_frequency(array(self.t), self.coupling)
        self.Ω["p"] = self.rabi_frequency(array(self.t), self.probe)

    def plot(self, save_directory=None):
        """Plots the results of a dark state simulation.

        Args:
            save_directory (str, optional): Directory to save the plot to. Defaults to None.
        """
        fig, ax1 = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(7)

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Population")
        ax1.plot(self.t, self.dcdt["x"], "k")
        ax1.plot(self.t, self.dcdt["y"], "b--")
        ax1.plot(self.t, self.dcdt["z"], "g.")
        ax1.tick_params(axis="y")
        ax1.legend([r"$c_{x}$", r"$c_{y}$", r"$c_{z}$"], loc=2)

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
            transfer = self.simulate_transfer(**kwargs)
            self.φ.append(mean(transfer["phase"]["x"]))

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

class PopulationTransfer(ElectromagneticallyInducedTransparency):
    def __init__(self, probe={"t": 6, "A": 5}, coupling={"t": 1, "A": None}):
        super().__init__(probe, coupling)
        self.x = arange(pi / 2, 3 * pi / 2, 0.001)
        self.max_amplitues = [10, 50]
        self.amplitudes = [[], []]
        self.amplitudes[0] = self.max_amplitues[0] * sin(self.x)
        self.amplitudes[1] = self.max_amplitues[1] * sin(self.x)
        self.φ = [[], []]

    def simulate(self, **kwargs):
        """Simulates the popultaion transfer of a system."""        
        for i, amplitudes in enumerate(self.amplitudes):
            for amplitude in amplitudes:
                self.coupling["A"] = amplitude
                transfer = self.simulate_transfer(**kwargs)
                self.φ[i].append(mean(transfer["population"]["z"]))

    def plot(self, save_directory=None):
        """Plots the results of a population transfer simulation.

        Args:
            save_directory (str, optional): Directory to save the plot to. Defaults to None.
        """        
        self.x = self.x / pi - 1
        fig, ax1 = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(7)

        ax1.plot(self.x, 2 * self.amplitudes[1] / self.max_amplitues[0], color="tab:red", linestyle="--")

        ax1.set_xlabel("x (microns)")
        ax1.set_ylabel(r"$Ω_{C}(x)/Ω_{P}$", color="black")
        ax1.plot(self.x, 2 * self.amplitudes[1] / self.max_amplitues[1], color="tab:blue", linestyle="--")
        ax1.tick_params(axis="y", labelcolor="black")

        ax2 = ax1.twinx()

        ax2.plot(self.x, self.φ[1], color="tab:red")

        ax2.set_ylabel("Population Transfer", color="black")
        ax2.plot(self.x, self.φ[0], color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="black")

        fig.tight_layout()

        if save_directory is not None:
            plt.savefig(f"{save_directory}/spatial_offset_vs_population_transfer.png")

        plt.show()


if __name__ == "__main__":
    DSS = DarkState()
    DSS.simulate()
    DSS.plot(save_directory=DSS.PLOT_DIR)

    PTS = PhaseTransfer()
    PTS.simulate()
    PTS.plot(save_directory=PTS.PLOT_DIR)

    PTS = PopulationTransfer()
    PTS.simulate()
    PTS.plot(save_directory=PTS.PLOT_DIR)
