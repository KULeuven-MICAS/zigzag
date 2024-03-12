import pdb
import math

class ADC:
    """
    Class for a single ADC.
    :param resolution: ADC resolution
    :param vdd: The supply vdd (unit: V)
    :param ICH: The number of input channels on bitline (ADC input node)
    """
    def __init__(self, resolution: int, ICH: int):
        self.resolution = resolution
        self.ICH = ICH
    def calculate_area(self):
        if self.resolution < 12:
            #self.area = 10 ** (-0.25 * self.resolution-3.3) * 2**self.resolution # unit: mm2
            self.area = (10**-6) * 10 ** (-0.0369 * self.resolution+1.206) * 2**self.resolution # unit: mm2
        else:
            self.area = 5 * 10**-7 * 2**self.resolution # unit: mm2
        return self.area
    def calculate_delay(self):
        self.delay = self.resolution * (0.00653*self.ICH+0.640) # ns
        return self.delay
    def calculate_energy(self, vdd): # unit: fJ
        k1 = 100 # fF
        k2 = 0.001 # fF
        self.energy = (k1 * self.resolution + k2 * 4**self.resolution) * vdd**2
        return self.energy
        
class DAC:
    """
    Class for a single DAC.
    :param resolution: DAC resolution
    :param vdd: The supply vdd (unit: V)
    """
    def __init__(self, resolution: int):
        self.resolution = resolution
    def calculate_area(self):
        self.area = 0
        return self.area
    def calculate_delay(self):
        self.delay = 0
        return self.delay
    def calculate_energy(self, vdd, k0): # unit: fF
        self.energy = (k0 * self.resolution) * vdd**2
        return self.energy