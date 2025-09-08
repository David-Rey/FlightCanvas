
import aerosandbox as asb
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

class FinTabs:
    def __init__(self):
        # Nose cone parameters
        self.nose_cone_length = 0.61  # [m]
        self.nose_cone_rho = 1  # []

        # Main body parameters
        self.total_length = 2.64  # This includes nose code [m]
        self.rocket_diameter = 0.157  # [m]

        # MOI and CG
        self.longitudinal_moi = 9.15  # [kg * m^2]
        self.rotational_moi = 0.112  # [kg * m^2]
        self.cg_x = 0  # [m]
        #self.cg_x = 1.84  # [m]

        # Fins
        self.num_fins = 4  # []
        self.root_chord = 0.203  # [m]
        self.tip_chord = 0.17  # [m]
        self.fin_height = 0.144  # [m]
        self.sweep_length = 0.033  # [m]
        self.fin_thickness = 0.00635

        body = self._create_body()
        fins = self._create_fins()

        # Assemble the AeroVehicle
        self.vehicle = asb.Airplane(
            name="FinTabs",
            xyz_ref=[0, 0, 0],
            wings=fins,
            fuselages=[body],
        )

    def _create_body(self) -> asb.Fuselage:
        """Creates the fuselage geometry for a rocket with a tangent ogive nose cone."""

        # Define key geometric parameters for clarity
        L = self.nose_cone_length  # Length of the nose cone
        R = self.rocket_diameter / 2  # Radius of the rocket body

        # Calculate the ogive radius (rho) based on L and R
        rho = (R ** 2 + L ** 2) / (2 * R)

        # Generate coordinates for the nose cone curve
        nose_x = np.linspace(0, L, 100)  # Use 100 points for a smooth curve

        # This is the standard equation for a tangent ogive's profile
        nose_y = np.sqrt(rho ** 2 - (L - nose_x) ** 2) + R - rho

        nose_coords = np.vstack([nose_x, nose_y]).T

        # Define the cylindrical body section
        body_coords = np.array([
            [self.total_length, R],
            [self.total_length, 0]
        ])

        # Combine all coordinates into a single profile
        full_body_coords = np.vstack((nose_coords, body_coords))

        # Create AeroSandbox FuselageXSec objects
        body_xsecs = [
            asb.FuselageXSec(
                xyz_c=[x - self.cg_x, 0, 0],
                radius=y,  # The y-coordinate from our profile is the radius
            ) for x, y in full_body_coords
        ]

        return asb.Fuselage(
            name="RocketBody",
            xsecs=body_xsecs
        )

    @staticmethod
    def _flat_plate_airfoil(thickness=0.01, n_points=100) -> np.ndarray:
        """
        Generate flat plate airfoil coordinates
        :param thickness: Percent thickness of flat plate airfoil
        :param n_points: Number of flat plate airfoil points
        """
        x = np.linspace(1, 0, n_points)
        y_upper = thickness / 2 * np.ones_like(x)
        y_lower = -thickness / 2 * np.ones_like(x)
        x_coords = np.concatenate([x, x[::-1]])
        y_coords = np.concatenate([y_upper, y_lower[::-1]])
        return np.vstack([x_coords, y_coords]).T

    def _create_fins(self) -> List[asb.Wing]:
        """Creates a set of 4 symmetric fins."""
        percent_thickness = self.fin_thickness / self.root_chord
        fin_airfoil = asb.Airfoil(coordinates=self._flat_plate_airfoil(thickness=percent_thickness))
        radius = self.rocket_diameter / 2

        # Define the leading edge x-position of the fin root
        # This is the absolute position from the nose tip
        fin_root_le_x_abs = self.total_length - self.root_chord

        # Define base coordinates for a single fin, relative to the CG
        root_le_base = np.array([fin_root_le_x_abs - self.cg_x, 0, 0])
        tip_le_base = np.array([fin_root_le_x_abs - self.cg_x + self.sweep_length, self.fin_height, 0])

        # Define the 4 fins by rotating the tip coordinates
        fins = []
        for i in range(4):
            angle_deg = i * 90
            angle_rad = np.deg2rad(angle_deg)

            # Rotate the tip coordinates
            tip_le_rotated = np.array([
                tip_le_base[0],
                tip_le_base[1] * np.cos(angle_rad),
                tip_le_base[1] * np.sin(angle_rad)
            ])

            # Determine the attachment point on the fuselage
            attach_point_rotated = np.array([
                0,  # x is handled by the root/tip coordinates
                radius * np.cos(angle_rad),
                radius * np.sin(angle_rad)
            ])

            fin = asb.Wing(
                name=f"Fin_{i + 1}",
                symmetric=False,
                xsecs=[
                    asb.WingXSec(xyz_le=root_le_base, chord=self.root_chord, airfoil=fin_airfoil),
                    asb.WingXSec(xyz_le=tip_le_rotated, chord=self.tip_chord, airfoil=fin_airfoil)
                ]
            ).translate(attach_point_rotated)
            fins.append(fin)

        return fins

    @staticmethod
    def _flat_plate_airfoil(thickness=0.01, n_points=100) -> np.ndarray:
        """
        Generate flat plate airfoil coordinates
        :param thickness: Percent thickness of flat plate airfoil
        :param n_points: Number of flat plate airfoil points
        """
        x = np.linspace(1, 0, n_points)
        y_upper = thickness / 2 * np.ones_like(x)
        y_lower = -thickness / 2 * np.ones_like(x)
        x_coords = np.concatenate([x, x[::-1]])
        y_coords = np.concatenate([y_upper, y_lower[::-1]])
        return np.vstack([x_coords, y_coords]).T

if __name__ == "__main__":
    rocket = FinTabs()


    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p
    import aerosandbox as asb
    import numpy as np

    # Assume the FinTabs class and rocket object are already defined
    # from your_file import FinTabs
    # rocket = FinTabs()

    # --- Analysis Code (from your example) ---
    # Note: A wider range like -90 to 90 can show strange results at the extremes
    # as the aerodynamic models break down. A more practical range is often used.
    alphas = np.linspace(-10, 10, 100)

    aero = asb.AeroBuildup(
        airplane=rocket.vehicle,
        op_point=asb.OperatingPoint(
            velocity=100,  # Using a more realistic velocity
            alpha=alphas,
            beta=0
        ),
    ).run_with_stability_derivatives()  # .run() is sufficient as it calculates x_np

    # --- Plotting Code ---
    # Extract the neutral point data
    x_np = aero["x_np"]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the neutral point location
    ax.plot(alphas, x_np, linewidth=2, label="Neutral Point (NP)")

    # Add a horizontal line for the Center of Gravity (CG)
    # This is crucial for stability analysis
    ax.axhline(
        y=1.84,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Center of Gravity (CG) = {rocket.cg_x:.2f} m"
    )

    # --- Formatting ---
    ax.set_xlabel("Angle of Attack [°]")
    ax.set_ylabel("Position [m from nose tip]")
    ax.set_title("Neutral Point and Static Stability vs. Angle of Attack")
    ax.legend()
    p.show_plot()

"""
    def show():
        p.set_ticks(15, 5, 15, 5)
        p.equal()
        p.show_plot(
            "`asb.AeroBuildup` Aircraft Aerodynamics",
            r"Sideslip angle $\beta$ [deg]",
            r"Angle of Attack $\alpha$ [deg]",
            set_ticks=False
        )


    fig, ax = plt.subplots(figsize=(6, 5))
    p.contour(
        Beta, Alpha, aero["CL"].reshape(Alpha.shape),
        colorbar_label="Lift Coefficient $C_L$ [-]",
        linelabels_format=lambda x: f"{x:.2f}",
        linelabels_fontsize=7,
        cmap="RdBu",
        alpha=0.6
    )
    plt.clim(*np.array([-1, 1]) * np.max(np.abs(aero["CL"])))
    show()

"""