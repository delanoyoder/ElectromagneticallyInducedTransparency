from os import environ
import matplotlib.pyplot as plt
from numpy import pi, tanh, array
from scipy.integrate import complex_ode


class Results:
    def __init__(self, key=None):
        self.key = key
        if key == "DSS":
            self.t = []
            self.c1 = []
            self.c2 = []
            self.c3 = []
            self.Ωc = []
            self.Ωp = []

class Plot:
    DIR = f"{environ['VIRTUAL_ENV'][:-5]}/plots"

    def plot(self, results):
        """Identifies the type of plot to be made and calls the appropriate function.

        Args:
            results (dict): Dictionary of results to be plotted.

        Raises:
            ValueError: If the results dictionary does not contain a valid key.
        """
        if results.key == "DSS":
            self.dark_state_simulation(results, save_directory=self.DIR)
        else:
            raise KeyError(f"Invalid key: {results.key}")

    @staticmethod
    def dark_state_simulation(results, save_directory=None):
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
        ax1.plot(results.t, results.c1, "k")
        ax1.plot(results.t, results.c2, "b--")
        ax1.plot(results.t, results.c3, "g.")
        ax1.tick_params(axis="y")
        ax1.legend([r"$c_{1}$", r"$c_{2}$", r"$c_{3}$"], loc=2)

        ax2 = ax1.twinx()

        ax2.set_ylabel(r"$Ω$ (arb. units)")
        ax2.plot(results.t, results.Ωc, "r:", alpha=0.5)
        ax2.plot(results.t, results.Ωp, "y-.", alpha=0.5)
        ax2.tick_params(axis="y")
        ax2.legend([r"$Ω_{C}$", r"$Ω_{P}$"], loc=1)

        fig.tight_layout()

        if save_directory is not None:
            plt.savefig(f"{save_directory}/dark_state_simulation.png")

        plt.show()


class ElectromagneticallyInducedTransparency:
    def __init__(self, probe={"t": pi, "A": 50}, coupling={"t": 0, "A": 50}):
        self.probe = probe
        self.coupling = coupling
        self.ode = complex_ode(self.differential_equation)

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

        c1 = (i / 2) * Ωp * c[2]
        c2 = (i / 2) * Ωc * c[2]
        c3 = (i / 2) * (Ωp * c[0] + Ωc * c[1])

        return [c1, c2, c3]

    def DarkStateSimulation(
        self,
        initial_state=[1, 0, 0],
        initial_time=0,
        final_time=10,
        dt=0.25,
        results=Results("DSS"),
    ):
        """Simulates the dark state of a system.

        Args:
            initial_state (list, optional): Initial state of the system. Defaults to [1, 0, 0].
            initial_time (int, optional): Initial time of the simulation. Defaults to 0.
            final_time (int, optional): Final time of the simulation. Defaults to 10.
            dt (float, optional): Time step. Defaults to 0.25.
            results (dict, optional): Dictionary of results. Defaults to defaultdict(list).

        Returns:
            dict: Dictionary of results.
        """    
        self.ode.set_initial_value(initial_state, initial_time)
        while self.ode.successful() and self.ode.t < final_time:
            self.ode.integrate(self.ode.t + dt)
            results.t.append(self.ode.t)
            results.c1.append(abs(self.ode.y[0]))
            results.c2.append(abs(self.ode.y[1]))
            results.c3.append(abs(self.ode.y[2]))

        results.Ωc = self.rabi_frequency(array(results.t), self.coupling)
        results.Ωp = self.rabi_frequency(array(results.t), self.probe)

        return results

    #########################
    # Static Methods ########
    #########################

    @staticmethod
    def rabi_frequency(t, laser):
        """Returns the Rabi frequency of a laser at a particular time.

        Args:
            t (float): Time.
            laser (dict): Dictionary of laser parameters.

        Returns:
            float: Rabi frequency.
        """        
        return laser["A"] * (tanh(t - laser["t"] - pi) + 1)


if __name__ == "__main__":
    EIT = ElectromagneticallyInducedTransparency()
    Plot().plot(EIT.DarkStateSimulation())
