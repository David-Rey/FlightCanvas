
class ComponentBuildup:
    def __init__(self, name: str,
                 vehicle_path: str,
                 aero_component: ['AeroComponent'],
                 alpha_grid_size: int = 150,
                 beta_grid_size: int = 100,
                 operating_velocity: float = 50.0,
                 ):

        self.name = name
        self.vehicle_path = vehicle_path
        self.aero_component = aero_component

        self.alpha_grid_size = alpha_grid_size
        self.beta_grid_size = beta_grid_size
        self.operating_velocity = operating_velocity

        self.alpha_grid = None  # Hold grid of an angle of attack for buildup
        self.beta_grid = None  # Hold grid of sideslip angles for buildup
        self.asb_data_static = None  # To hold the aero build data
        self.aero_interpolants = None  # To hold the CasADi interpolant object

        


if __name__ == '__main__':
    print(1)