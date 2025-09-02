import pyvista as pv
import numpy as np


class ThrustVisualizer:
    def __init__(self, rocket_position=(0, 0, 0), nozzle_radius=0.8):
        self.rocket_pos = np.array(rocket_position)
        self.nozzle_radius = nozzle_radius
        self.thrust_level = 0.5  # 0 = no thrust, 1 = max thrust
        self.plotter = None
        self.exhaust_actors = []

    def create_rocket_mesh(self):
        """Create a simple rocket body"""
        # Main body (cylinder)
        body = pv.Cylinder(center=self.rocket_pos + [0, 0, 2],
                           radius=0.5, height=4, direction=[0, 0, 1])

        # Nose cone
        nose = pv.Cone(center=self.rocket_pos + [0, 0, 4.5],
                       radius=0.5, height=1, direction=[0, 0, 1])

        # Nozzle
        nozzle = pv.Cone(center=self.rocket_pos + [0, 0, -0.5],
                         radius=self.nozzle_radius, height=1,
                         direction=[0, 0, -1])

        return body + nose + nozzle

    def create_exhaust_plume(self, thrust_level):
        """Create exhaust plume based on thrust level"""
        if thrust_level <= 0:
            return []  # No exhaust

        exhaust_meshes = []

        # Scale exhaust size with thrust
        max_length = 8.0
        max_radius = 2.0

        plume_length = max_length * thrust_level
        plume_radius = max_radius * thrust_level

        # Create multiple cones for a more realistic exhaust shape
        num_sections = 5

        for i in range(num_sections):
            # Each section gets progressively larger and more transparent
            section_progress = i / (num_sections - 1)

            # Position along exhaust length
            z_pos = -1.5 - (plume_length * section_progress)

            # Radius grows with distance from nozzle
            radius = self.nozzle_radius + (plume_radius * section_progress)

            # Height of this section
            section_height = plume_length / num_sections * 1.2  # Slight overlap

            # Create cone section
            cone = pv.Cone(
                center=self.rocket_pos + [0, 0, z_pos - section_height / 2],
                radius=radius,
                height=section_height,
                direction=[0, 0, -1],
                resolution=20
            )

            # Color: hot at nozzle (white/blue) to cooler at end (red/orange)
            if section_progress < 0.3:
                # Hot core - white to light blue
                color = [1.0, 0.9 + 0.1 * section_progress, 1.0]
            elif section_progress < 0.7:
                # Transition - blue to yellow
                t = (section_progress - 0.3) / 0.4
                color = [1.0, 0.7 + 0.3 * t, 1.0 - 0.5 * t]
            else:
                # Cool outer - yellow to red
                t = (section_progress - 0.7) / 0.3
                color = [1.0, 0.6 - 0.4 * t, 0.2 - 0.2 * t]

            # Opacity decreases with distance and lower thrust
            opacity = (1.0 - section_progress * 0.7) * thrust_level * 0.8

            exhaust_meshes.append({
                'mesh': cone,
                'color': color,
                'opacity': opacity
            })

        return exhaust_meshes

    def update_thrust(self, new_thrust_level):
        """Update thrust level and refresh visualization"""
        self.thrust_level = max(0.0, min(1.0, new_thrust_level))

        if self.plotter is not None:
            self.refresh_exhaust()

    def refresh_exhaust(self):
        """Remove old exhaust and add new based on current thrust"""
        if self.plotter is None:
            return

        # Remove existing exhaust actors
        for actor_name in self.exhaust_actors:
            try:
                self.plotter.remove_actor(actor_name)
            except:
                pass
        self.exhaust_actors.clear()

        # Add new exhaust based on current thrust
        exhaust_sections = self.create_exhaust_plume(self.thrust_level)

        for i, section in enumerate(exhaust_sections):
            actor_name = f"exhaust_section_{i}"
            self.plotter.add_mesh(
                section['mesh'],
                color=section['color'],
                opacity=section['opacity'],
                name=actor_name,
                smooth_shading=True
            )
            self.exhaust_actors.append(actor_name)

    def create_visualization(self):
        """Create the complete visualization"""
        self.plotter = pv.Plotter(window_size=(1200, 800))

        # Add rocket
        rocket = self.create_rocket_mesh()
        self.plotter.add_mesh(rocket, color='silver', metallic=0.8, roughness=0.2)

        # Set up scene
        self.plotter.set_background('black')
        self.plotter.camera.position = (8, -12, 4)
        self.plotter.camera.focal_point = (0, 0, 0)

        # Add initial exhaust
        self.refresh_exhaust()

        # Add title and thrust indicator
        self.plotter.add_text("Rocket Engine Thrust Visualization",
                              position='upper_left', font_size=14, color='white')

        return self.plotter


def create_interactive_thrust_demo():
    """Create interactive demo with thrust controls"""

    # Create thrust visualizer
    thrust_viz = ThrustVisualizer()
    plotter = thrust_viz.create_visualization()

    # Add thrust level display
    thrust_text = plotter.add_text(
        f"Thrust Level: {thrust_viz.thrust_level:.1%}",
        position='upper_right', font_size=12, color='white'
    )

    # Keyboard callback for thrust control
    def update_thrust_callback():
        def callback(obj, event):
            key = obj.GetKeySym()

            current_thrust = thrust_viz.thrust_level

            if key == 'Up':
                new_thrust = min(1.0, current_thrust + 0.1)
            elif key == 'Down':
                new_thrust = max(0.0, current_thrust - 0.1)
            elif key == '0':
                new_thrust = 0.0
            elif key == '1':
                new_thrust = 0.25
            elif key == '2':
                new_thrust = 0.5
            elif key == '3':
                new_thrust = 0.75
            elif key == '4':
                new_thrust = 1.0
            else:
                return

            # Update thrust
            thrust_viz.update_thrust(new_thrust)

            # Update text display
            thrust_text.SetInput(f"Thrust Level: {thrust_viz.thrust_level:.1%}")
            plotter.render()

        return callback

    # Add key event observer
    plotter.iren.AddObserver('KeyPressEvent', update_thrust_callback())

    # Add instructions
    instructions = (
        "Controls:\n"
        "↑/↓ arrows: Adjust thrust\n"
        "0-4 keys: Set thrust levels\n"
        "0=Off, 1=25%, 2=50%, 3=75%, 4=100%"
    )

    plotter.add_text(instructions, position='lower_left',
                     font_size=10, color='lightgray')

    # Show the visualization
    plotter.show()


def create_thrust_comparison():
    """Create side-by-side comparison of different thrust levels"""

    plotter = pv.Plotter(shape=(1, 4), window_size=(1600, 400))

    thrust_levels = [0.25, 0.5, 0.75, 1.0]
    titles = ["25% Thrust", "50% Thrust", "75% Thrust", "100% Thrust"]

    for i, (thrust, title) in enumerate(zip(thrust_levels, titles)):
        plotter.subplot(0, i)

        # Create visualizer for this thrust level
        viz = ThrustVisualizer(rocket_position=(0, 0, 0))
        viz.thrust_level = thrust

        # Add rocket
        rocket = viz.create_rocket_mesh()
        plotter.add_mesh(rocket, color='silver')

        # Add exhaust
        exhaust_sections = viz.create_exhaust_plume(thrust)
        for j, section in enumerate(exhaust_sections):
            plotter.add_mesh(
                section['mesh'],
                color=section['color'],
                opacity=section['opacity']
            )

        # Set up this subplot
        plotter.set_background('black')
        plotter.camera.position = (6, -8, 3)
        plotter.camera.focal_point = (0, 0, 0)
        plotter.add_text(title, position='upper_left', font_size=12)

    plotter.show()


if __name__ == "__main__":
    print("Choose visualization:")
    print("1. Interactive thrust control")
    print("2. Thrust level comparison")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        create_interactive_thrust_demo()
    else:
        create_thrust_comparison()