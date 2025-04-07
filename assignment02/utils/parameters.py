from config import config


class Parameters:
    def __init__(self):
        # Given parameters
        self.m = config.parameters.m          # Max chargers per station
        self.q = config.parameters.q          # Max robots per charger
        self.c_b = config.parameters.c_b      # Investment cost per station
        self.c_h = config.parameters.c_h      # Cost of moving a robot
        self.c_m = config.parameters.c_m      # Maintenance cost per charger
        self.c_c = config.parameters.c_c      # Charging cost per km
        self.ld = config.parameters.ld        # Lambda parameter for exponential distribution
        self.r_min = config.parameters.r_min  # Minimum range of a robot
        self.r_max = config.parameters.r_max  # Maximum range of a robot
