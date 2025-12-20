import numpy as np


class Log:
    def __init__(self, state_names: list, deflection_names: list, max_steps: int):
        """
        Initialize the logger with preallocated NumPy arrays.
        """
        self.state_names = list(state_names)
        self.state_dot_names = list(state_names)
        self.ekf_covariance_names = ['Px', 'Py', 'Pz', 'Vx', 'Vy', 'Vz']
        self.deflection_names = list(deflection_names)
        self.aero_forces_names = ['Fx', 'Fy', 'Fz']
        self.aero_moments_names = ['Mx', 'My', 'Mz']

        self.max_steps = max_steps
        self.current_idx = -1  # Python uses 0-based indexing

        # Preallocate arrays
        self.time = np.zeros(self.max_steps)
        self.states = np.zeros((len(self.state_names), self.max_steps))
        self.state_dots = np.zeros((len(self.state_dot_names), self.max_steps))
        self.state_estimates = np.zeros((len(self.state_names), self.max_steps))
        self.ekf_covariances = np.zeros((len(self.ekf_covariance_names), self.max_steps))
        self.control_inputs = np.zeros((len(self.deflection_names), self.max_steps))
        self.aero_forces = np.zeros((3, self.max_steps))
        self.aero_moments = np.zeros((3, self.max_steps))

    def initialize_timestep(self, t: float):
        """Advances index and stores the current time."""
        self.current_idx += 1

        if self.current_idx >= self.max_steps:
            print("Warning: Logger capacity exceeded. Expanding arrays...")
            self.expand()

        self.time[self.current_idx] = t

    def add(self, t: float, attribute_name: str, value):
        """Adds data to the specified attribute for the current timestep."""
        if self.current_idx < 0:
            raise RuntimeError("Must call initialize_timestep(t) before adding data")

        if abs(self.time[self.current_idx] - t) > 1e-10:
            raise ValueError(f"Time mismatch: expected t={self.time[self.current_idx]:.6f}, got t={t:.6f}")

        attr = attribute_name.lower()
        value = np.array(value).flatten()  # Ensure it's a flat array

        # Validation and assignment logic
        if attr == 'states':
            self._validate_size(value, len(self.state_names), attr)
            self.states[:, self.current_idx] = value
        elif attr == 'state_dots':
            self._validate_size(value, len(self.state_dot_names), attr)
            self.state_dots[:, self.current_idx] = value
        elif attr == 'state_estimates':
            self._validate_size(value, len(self.state_names), attr)
            self.state_estimates[:, self.current_idx] = value
        elif attr == 'ekf_covariances':
            self._validate_size(value, len(self.ekf_covariance_names), attr)
            self.ekf_covariances[:, self.current_idx] = value
        elif attr == 'control_inputs':
            self._validate_size(value, len(self.deflection_names), attr)
            self.control_inputs[:, self.current_idx] = value
        elif attr == 'aero_forces':
            self._validate_size(value, 3, attr)
            self.aero_forces[:, self.current_idx] = value
        elif attr == 'aero_moments':
            self._validate_size(value, 3, attr)
            self.aero_moments[:, self.current_idx] = value
        else:
            raise KeyError(f"Unknown attribute '{attribute_name}'")

    @staticmethod
    def _validate_size(value: np.ndarray, expected: int, name: str):
        if len(value) != expected:
            raise ValueError(f"{name} size mismatch: expected {expected}, got {len(value)}")

    def trim(self):
        """Removes unused preallocated space."""
        valid_range = slice(0, self.current_idx + 1)
        self.time = self.time[valid_range]
        self.states = self.states[:, valid_range]
        self.state_dots = self.state_dots[:, valid_range]
        self.state_estimates = self.state_estimates[:, valid_range]
        self.ekf_covariances = self.ekf_covariances[:, valid_range]
        self.control_inputs = self.control_inputs[:, valid_range]
        self.aero_forces = self.aero_forces[:, valid_range]
        self.aero_moments = self.aero_moments[:, valid_range]

    def expand(self):
        """Doubles the preallocated capacity."""
        new_steps = self.max_steps
        extra = np.zeros_like(self.time)  # Create zero buffers of current size

        self.time = np.concatenate([self.time, np.zeros(new_steps)])
        self.states = np.hstack([self.states, np.zeros((self.states.shape[0], new_steps))])
        self.state_dots = np.hstack([self.state_dots, np.zeros((self.state_dots.shape[0], new_steps))])
        self.state_estimates = np.hstack([self.state_estimates, np.zeros((self.state_estimates.shape[0], new_steps))])
        self.ekf_covariances = np.hstack([self.ekf_covariances, np.zeros((self.ekf_covariances.shape[0], new_steps))])
        self.control_inputs = np.hstack([self.control_inputs, np.zeros((self.control_inputs.shape[0], new_steps))])
        self.aero_forces = np.hstack([self.aero_forces, np.zeros((3, new_steps))])
        self.aero_moments = np.hstack([self.aero_moments, np.zeros((3, new_steps))])

        self.max_steps *= 2

    def _get_by_name(self, names_list: list, data_matrix: np.ndarray, name: str) -> np.ndarray:
        try:
            idx = names_list.index(name)
            return data_matrix[idx, :self.current_idx + 1]
        except ValueError:
            raise ValueError(f"Name '{name}' not found in list.")

    def get_state(self, name: str) -> np.ndarray:
        return self._get_by_name(self.state_names, self.states, name)

    def get_control_input(self, name: str) -> np.ndarray:
        return self._get_by_name(self.deflection_names, self.control_inputs, name)
