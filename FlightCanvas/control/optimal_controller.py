
from FlightCanvas.control.controller import BaseController
from FlightCanvas.vehicle.aero_vehicle import AeroVehicle

class OptimalController(BaseController):
    def __init__(self, vehicle: AeroVehicle):
        super().__init__()

        self._vehicle = vehicle

