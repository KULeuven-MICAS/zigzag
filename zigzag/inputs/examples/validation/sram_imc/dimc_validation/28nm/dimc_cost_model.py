import pdb
import math

class UnitNor2:
    """
    Class for a single NOR2 gate.
    Currently, the data is from foundry. It should be replaced by extracted data for open-source purpose.
    :param unit_area: The area cost (unit: mm2)
    :param unit_delay: The delay cost (unit: ns)
    :param unit_cap: The input capacitance including all input ports (unit: fF)
    """
    def __init__(self, unit_area: float, unit_delay: float, unit_cap: float):
        self.area = unit_area/(10**6)
        self.delay = unit_delay
        self.cap = unit_cap
    def calculate_area(self):
        return self.area
    def calculate_delay(self):
        return self.delay
    def calculate_cap(self):
        return self.cap
    
    
class UnitNand2:
    """
    Class for a single NAND2 gate.
    Currently, the data is from foundry. It should be replaced by extracted data for open-source purpose.
    :param unit_area: The area cost (unit: mm2)
    :param unit_delay: The delay cost (unit: ns)
    :param unit_cap: The input capacitance including all input ports (unit: fF)
    """
    def __init__(self, unit_area: float, unit_delay: float, unit_cap: float):
        self.area = unit_area/(10**6)
        self.delay = unit_delay
        self.cap = unit_cap
    def calculate_area(self):
        return self.area
    def calculate_delay(self):
        return self.delay
    def calculate_cap(self):
        return self.cap
    
    
class UnitXor2:
    """
    Class for a single XOR2 gate.
    Currently, the data is from foundry. It should be replaced by extracted data for open-source purpose.
    :param unit_area: The area cost (unit: mm2)
    :param unit_delay: The delay cost (unit: ns)
    :param unit_cap: The input capacitance including all input ports (unit: fF)
    """
    def __init__(self, unit_area: float, unit_delay: float, unit_cap: float):
        self.area = unit_area*2.4/(10**6)
        self.delay = unit_delay*2.4
        self.cap = unit_cap*1.5
    def calculate_area(self):
        return self.area
    def calculate_delay(self):
        return self.delay
    def calculate_cap(self):
        return self.cap
    
    
class UnitDff:
    """
    Class for a single 1-b DFF.
    Currently, the data is from foundry. It should be replaced by extracted data for open-source purpose.
    :param unit_area: The area cost (unit: mm2)
    :param unit_delay: The delay cost (unit: ns)
    :param unit_cap: The input capacitance including all input ports (unit: fF)
    """
    def __init__(self, unit_area: float, unit_delay: float, unit_cap: float):
        self.area = unit_area*6/(10**6)
        self.delay = 0
        self.cap = unit_cap*3
    def calculate_area(self):
        return self.area
    def calculate_delay(self):
        return self.delay
    def calculate_cap(self):
        return self.cap
###############################################################################################################
class Multiplier:
    def __init__(self, vdd: float, input_precision: int, unit_area: float, unit_delay: float, unit_cap: float):
        """
        Class for a single multiplier that performs 1 bit x multiple bits
        :param vdd:              The supply voltage (unit: V)
        :param input_precision:  The bit precision of the input (unit: bit)
        :param output_precision: The bit precision of the output (unit: bit)
        """
        self.nor2 = UnitNor2(unit_area, unit_delay, unit_cap)
        self.vdd = vdd
        self.input_precision = input_precision
        self.output_precision = input_precision # output precision = input precision
    
    def calculate_area(self):
        """
        area: The area cost (unit: mm2)
        """
        area = self.nor2.calculate_area() * self.input_precision
        return area
        
    def calculate_delay(self):
        """
        delay: The delay cost (unit: ns)
        """
        delay = self.nor2.calculate_delay()
        return delay
        
    def calculate_energy(self):
        """
        energy: The energy cost (unit: fJ)
        """
        energy = self.nor2.calculate_cap()/2 * self.vdd**2 * self.input_precision # /2 is because only input will change, weight doesn't change
        return energy

class MultiplierArray:
    def __init__(self, vdd: float, input_precision: int, number_of_multiplier: int, unit_area: float, unit_delay: float, unit_cap: float):
        """
        Class for a single multiplier that performs 1 bit x multiple bits
        :param vdd:                  The supply voltage (unit: V)
        :param input_precision:      The bit precision of the input (unit: bit)
        :param output_precision:     The bit precision of the output (unit: bit)
        :param number_of_multiplier: The number of multiplier
        """
        self.mult = Multiplier(vdd, input_precision, unit_area, unit_delay, unit_cap)
        self.vdd = vdd
        self.input_precision = input_precision
        self.output_precision = input_precision # output precision = input precision
        self.number_of_multiplier = number_of_multiplier
    
    def calculate_area(self):
        """
        area: The area cost (unit: mm2)
        """
        area = self.mult.calculate_area() * self.number_of_multiplier
        return area
        
    def calculate_delay(self):
        """
        delay: The delay cost (unit: ns)
        """
        delay = self.mult.calculate_delay()
        return delay
        
    def calculate_energy(self):
        """
        energy: The energy cost (unit: fJ)
        """
        energy = self.mult.calculate_energy() * self.number_of_multiplier
        return energy


class Adder:
    def __init__(self, vdd: float, input_precision: int, unit_area: float, unit_delay: float, unit_cap: float):
        """
        Class for a {input_precision}-b Carry-Ripple Adder
        :param vdd:                The supply voltage (unit: V)
        :param input_precision:    The bit precision of the input (unit: bit)
        :param output_precision:   The bit precision of the output (unit: bit)
        :param number_of_1b_adder: The number of 1-b adder in the adder tree
        """
        self.nand2 = UnitNand2(unit_area, unit_delay, unit_cap)
        self.xor2 = UnitXor2(unit_area, unit_delay, unit_cap)
        self.vdd = vdd
        self.input_precision = input_precision
        self.output_precision = input_precision + 1
        self.number_of_1b_adder = input_precision
        
    def calculate_area(self):
        """
        area: The area cost (unit: mm2)
        """
        area = (3*self.nand2.calculate_area() + 2*self.xor2.calculate_area())*self.number_of_1b_adder
        return area
    
    def calculate_delay_lsb(self):
        """
        delay: The delay cost for LSB (unit: ns) (best-case delay, also equals to the delay for Tsum of 1-b adder)
        """
        delay_sum = 2*self.xor2.calculate_delay() # 2 XOR gate delay (A-to-Sum)
        return delay_sum
    
    def calculate_delay_msb(self):
        """
        delay: The delay cost for MSB (unit: ns) (worst-case delay)
        """
        delay_carry = (self.xor2.calculate_delay() + 2*self.nand2.calculate_delay()) + (2*self.nand2.calculate_delay()) * (self.input_precision-1) # A-to-Cout -> Cin-to-Count * (precision-1)
        return delay_carry
    
    def calculate_energy(self):
        """
        energy: The energy cost (each time it is triggered) (unit: fJ)
        """
        energy = (2*self.xor2.calculate_cap() + 3*self.nand2.calculate_cap()) * self.vdd**2 * self.number_of_1b_adder
        return energy
        

class AdderTree:
    def __init__(self, vdd: float, input_precision: int, number_of_input: int, unit_area: float, unit_delay: float, unit_cap: float):
        """
        Class for a {input_number} {input_precision}-b Carry-Ripple Adder Tree
        :param vdd:                The supply voltage (unit: V)
        :param input_precision:    The bit precision of the input (unit: bit)
        :param number_of_input:    The number of inputs
        :param output_precision:   The bit precision of the output (unit: bit)
        :param number_of_1b_adder: The number of 1-b adder in the adder tree
        """
        if(math.log(number_of_input,2)%1 != 0):
            raise ValueError("The number of input for the adder tree is not in the power of 2. Currently it is: %s" %number_of_input)
        self.vdd = vdd
        self.input_precision = input_precision
        self.number_of_input = number_of_input
        self.depth = int( math.log(number_of_input, 2) )
        self.output_precision = input_precision + self.depth
        self.number_of_1b_adder = number_of_input*(input_precision+1)-(input_precision+self.depth+1)
        self.unit_area = unit_area
        self.unit_delay = unit_delay
        self.unit_cap = unit_cap
        
    def calculate_area(self):
        """
        area: The area cost (unit: mm2)
        """
        # calculate area iteratively
        # area_b = 0
        # for stage_idx in range(0, self.depth):
        #     single_adder = Adder(self.vdd, self.input_precision+stage_idx)
        #     area_b += single_adder.calculate_area() * math.ceil( self.number_of_input/(2**(stage_idx+1)) )
        # calculate area directly
        area = self.number_of_1b_adder * Adder(vdd=self.vdd, input_precision=1, unit_area=self.unit_area, unit_delay=self.unit_delay, unit_cap=self.unit_cap).calculate_area()
        return area
    
    def calculate_delay(self):
        """
        delay: The delay cost (unit: ns)
        """
        last_adder = Adder(vdd=self.vdd, input_precision=self.output_precision-1, unit_area=self.unit_area, unit_delay=self.unit_delay, unit_cap=self.unit_cap)
        delay = last_adder.calculate_delay_lsb() * (self.depth-1) + last_adder.calculate_delay_msb()
        return delay
    
    def calculate_energy(self):
        """
        energy: The energy cost (each time it is triggered) (unit: fJ)
        """
        energy = self.number_of_1b_adder * Adder(vdd=self.vdd, input_precision=1, unit_area=self.unit_area, unit_delay=self.unit_delay, unit_cap=self.unit_cap).calculate_energy()
        return energy
        
        


class MemoryInstance:
    """
    class for: regs (input regs, otuput regs), memory bank (copy from Zigzag code, with area, delay added)
    """
    def __init__(self, name: str, size: int, r_bw: int, w_bw: int, delay: float, r_energy: float, w_energy: float, area: float,
                 r_port: int=1, w_port: int=1, rw_port: int=0, latency: int=1,
                 min_r_granularity=None, min_w_granularity=None):
        """
        Collect all the basic information of a physical memory module.
        :param name: memory module name, e.g. 'SRAM_512KB_BW_16b', 'I_RF'
        :param size: total memory capacity (unit: bit)
        :param r_bw/w_bw: memory bandwidth (or wordlength) (unit: bit/cycle)
        :param delay: clock-to-output delay (unit: ns)
        :param r_energy/w_energy: memory unit data access energy (unit: fJ)
        :param area: memory area (unit: mm2)
        :param r_port: number of memory read port
        :param w_port: number of memory write port (rd_port and wr_port can work in parallel)
        :param rw_port: number of memory port for both read and write (read and write cannot happen in parallel)
        :param latency: memory access latency (unit: number of cycles)
        """
        self.name = name
        self.size = size
        self.r_bw = r_bw
        self.w_bw = w_bw
        self.delay = delay
        self.r_energy = r_energy
        self.w_energy = w_energy
        self.area = area
        self.r_port = r_port
        self.w_port = w_port
        self.rw_port = rw_port
        self.latency = latency
        if not min_r_granularity:
            self.r_bw_min = r_bw
        else:
            self.r_bw_min = min_r_granularity
        if not min_w_granularity:
            self.w_bw_min = w_bw
        else:
            self.w_bw_min = min_w_granularity

    def __jsonrepr__(self):
        """
        JSON Representation of this class to save it to a json file.
        """
        return self.__dict__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MemoryInstance) and self.__dict__ == other.__dict__

    
################
    
