
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import aerosandbox as asb
import aerosandbox.numpy as np

wing_airfoil = asb.Airfoil("sd7037")
tail_airfoil = asb.Airfoil("naca0010")

airplane = asb.Airplane(
    name="Peter's Glider",
    xyz_ref=[0, 0, 0],  # CG location
    wings=[
        asb.Wing(
            name="Main Wing",
            symmetric=True,  # Should this wing be mirrored across the XZ plane?
            xsecs=[  # The wing's cross ("X") sections
                asb.WingXSec(  # Root
                    xyz_le=[0, 0, 0],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                    chord=0.18,
                    twist=2,  # degrees
                    airfoil=wing_airfoil,  # Airfoils are blended between a given XSec and the next one.
                ),
                asb.WingXSec(  # Mid
                    xyz_le=[0.01, 0.5, 0],
                    chord=0.16,
                    twist=0,
                    airfoil=wing_airfoil,
                ),
                asb.WingXSec(  # Tip
                    xyz_le=[0.08, 1, 0.1],
                    chord=0.08,
                    twist=-2,
                    airfoil=wing_airfoil,
                ),
            ]
        ),
        asb.Wing(
            name="Horizontal Stabilizer",
            symmetric=True,
            xsecs=[
                asb.WingXSec(  # root
                    xyz_le=[0, 0, 0],
                    chord=0.1,
                    twist=-10,
                    airfoil=tail_airfoil,
                ),
                asb.WingXSec(  # tip
                    xyz_le=[0.02, 0.17, 0],
                    chord=0.08,
                    twist=-10,
                    airfoil=tail_airfoil
                )
            ]
        ).translate([0.6, 0, 0.06]),
        asb.Wing(
            name="Vertical Stabilizer",
            symmetric=False,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.1,
                    twist=0,
                    airfoil=tail_airfoil,
                ),
                asb.WingXSec(
                    xyz_le=[0.04, 0, 0.15],
                    chord=0.06,
                    twist=0,
                    airfoil=tail_airfoil
                )
            ]
        ).translate([0.6, 0, 0.07])
    ],
    fuselages=[
        asb.Fuselage(
            name="Fuselage",
            xsecs=[
                asb.FuselageXSec(
                    xyz_c=[0.8 * xi - 0.1, 0, 0.1 * xi - 0.03],
                    radius=0.6 * asb.Airfoil("dae51").local_thickness(x_over_c=xi)
                )
                for xi in np.cosspace(0, 1, 30)
            ]
        )
    ]
)

Alpha, Beta = np.meshgrid(np.linspace(-45, 45, 75), np.linspace(-85, 85, 100))
aero = asb.AeroBuildup(
    airplane=airplane,
    op_point=asb.OperatingPoint(
        velocity=50,
        alpha=Alpha.flatten(),
        beta=Beta.flatten()
    ),
).run_with_stability_derivatives(alpha=False, beta=False)

print(1)
def show():
    p.set_ticks(15, 5, 15, 5)
    p.equal()
    p.show_plot(
        "`asb.AeroBuildup` Aircraft Aerodynamics",
        r"Alpha [deg]",
        r"Beta [deg]",
        set_ticks=False
    )

'''
fig, ax = plt.subplots(figsize=(6, 5))
p.contour(
    Beta, Alpha, aero["Clp"].reshape(Alpha.shape),
    colorbar_label="Lift Coefficient $C_l$ p (roll) [-]",
    linelabels_format=lambda x: f"{x:.2f}",
    linelabels_fontsize=7,
    cmap="RdBu",
    alpha=0.6
)
plt.clim(*np.array([-1, 1]) * np.max(np.abs(aero["Clp"])))
show()

fig, ax = plt.subplots(figsize=(6, 5))
p.contour(
    Beta, Alpha, aero["Clq"].reshape(Alpha.shape),
    colorbar_label="Lift Coefficient $C_l$ q (ptich) [-]",
    linelabels_format=lambda x: f"{x:.2f}",
    linelabels_fontsize=7,
    cmap="RdBu",
    alpha=0.6
)
plt.clim(*np.array([-1, 1]) * np.max(np.abs(aero["Clq"])))
show()
'''
fig, ax = plt.subplots(figsize=(6, 5))
p.contour(
    Beta, Alpha, aero["Cnr"].reshape(Alpha.shape),
    colorbar_label="Lift Coefficient $C_n$ r (yaw) [-]",
    linelabels_format=lambda x: f"{x:.2f}",
    linelabels_fontsize=7,
    cmap="RdBu",
    alpha=0.6
)
plt.clim(*np.array([-1, 1]) * np.max(np.abs(aero["Cnr"])))
show()

"""
fig, ax = plt.subplots(figsize=(6, 5))
p.contour(
    Beta, roll, aero["CD"].reshape(roll.shape),
    colorbar_label="Drag Coefficient $C_D$ [-]",
    linelabels_format=lambda x: f"{x:.2f}",
    linelabels_fontsize=7,
    z_log_scale=True,
    cmap="YlOrRd",
    alpha=0.6
)
show()

fig, ax = plt.subplots(figsize=(6, 5))
p.contour(
    Beta, roll, (aero["CL"] / aero["CD"]).reshape(roll.shape),
    levels=15,
    colorbar_label="Finesse $C_L / C_D$ [-]",
    linelabels_format=lambda x: f"{x:.0f}",
    linelabels_fontsize=7,
    cmap="RdBu",
    alpha=0.6
)
plt.clim(*np.array([-1, 1]) * np.max(np.abs(aero["CL"] / aero["CD"])))
show()
"""