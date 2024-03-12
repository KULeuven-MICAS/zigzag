import pdb
from dimc_cost_model import *

def dimc_cost_estimation4(dimc, cacti_value):
    unit_reg = UnitDff(dimc['unit_area'], dimc['unit_delay'], dimc['unit_cap'])
    unit_area = dimc['unit_area']
    unit_delay = dimc['unit_delay']
    unit_cap = dimc['unit_cap']
    input_channel = dimc['input_channel']
    """
    multiplier array for each output channel
    """
    mults = MultiplierArray(vdd=dimc['vdd'],input_precision=int(dimc['multiplier_precision']),number_of_multiplier=input_channel*dimc['input_precision'], unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
    
    """
    adder_tree (1/3) for each output channel
    """
    
    adder_tree1 = AdderTree(vdd=dimc['vdd'], input_precision=1, number_of_input=16, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
    adder_tree2 = AdderTree(vdd=dimc['vdd'], input_precision=4, number_of_input=8, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
    adder_tree3 = AdderTree(vdd=dimc['vdd'], input_precision=6, number_of_input=8, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
   
    """
    accumulator for each output channel
    """
    accumulator = Adder(vdd=dimc['vdd'], input_precision=dimc['accumulator_precision'], unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)

    """
    memory instance (delay unit: ns, energy unit: fJ, area unit: mm2)
    unitbank: sram bank, data from CACTI
    """
    unitbank = MemoryInstance(name='unitbank', size=dimc['rows']*dimc['cols'], r_bw=dimc['cols'], w_bw=dimc['cols'], delay=cacti_value['delay']*0, r_energy=cacti_value['r_energy'], w_energy=cacti_value['w_energy'], area=cacti_value['area'], r_port=1, w_port=1, rw_port=0, latency=0)
    regs_accumulator = MemoryInstance(name='regs_accumulator', size=dimc['reg_accumulator_precision'], r_bw=dimc['reg_accumulator_precision'], w_bw=dimc['reg_accumulator_precision'], delay=unit_reg.calculate_delay(), r_energy=0, w_energy=unit_reg.calculate_cap() * dimc['vdd']**2 * dimc['reg_accumulator_precision'], area=unit_reg.calculate_area()*dimc['reg_accumulator_precision'], r_port=1, w_port=1, rw_port=0, latency=0)
    regs_pipeline = MemoryInstance(name='regs_accumulator', size=dimc['reg_pipeline_precision'], r_bw=dimc['reg_pipeline_precision'], w_bw=dimc['reg_pipeline_precision'], delay=unit_reg.calculate_delay(), r_energy=0, w_energy=unit_reg.calculate_cap() * dimc['vdd']**2 * dimc['reg_pipeline_precision'], area=unit_reg.calculate_area()*dimc['reg_pipeline_precision'], r_port=1, w_port=1, rw_port=0, latency=0)

    """
    calculate result
    :predicted_area:                The area cost for entire IMC core (unit: mm2)
    :predicted_delay:               The minimum delay of single clock period (unit: ns)
    :predicted_energy_per_cycle:    The energy cost each time the IMC core is activated (unit: fJ)
    :number_of_cycle: The number of cycle for computing entire input
    :predicted_energy:              The energy cost for computing entire input (unit: fJ)
    :number_of_operations:          The number of operations executed when computing entire input
    :predicted_tops:                Peak TOP/s
    :predicted_topsw:               Peak TOP/s/W
    """
    
    ## Area cost breakdown
    area_mults = dimc['banks'] * dimc['output_channel'] * mults.calculate_area()
    area_adder_tree = dimc['banks'] * dimc['output_channel'] * ( 8*8*adder_tree1.calculate_area() + 8*adder_tree2.calculate_area() + adder_tree3.calculate_area() )
    area_regs_pipeline = dimc['banks'] * dimc['output_channel'] * 8*regs_pipeline.area
    area_accumulator = dimc['banks'] * dimc['output_channel'] * accumulator.calculate_area()
    area_banks = dimc['banks'] * unitbank.area
    area_regs_accumulator = dimc['banks'] * dimc['output_channel'] * regs_accumulator.area
    
    predicted_area = area_mults + area_adder_tree + area_regs_pipeline + area_accumulator + area_banks + area_regs_accumulator # cost of input/output regs is not taken out
    
    ## Minimum clock time
    adder_1b_carry_delay = 2*UnitNand2(unit_area, unit_delay, unit_cap).calculate_delay()
    accumulator_delay = accumulator.calculate_delay_lsb()+adder_1b_carry_delay * (dimc['reg_accumulator_precision']-dimc['accumulator_input_precision'])
    #predicted_delay = max(unitbank.delay + mults.calculate_delay() + adder_tree1.calculate_delay() + adder_tree2.calculate_delay(), adder_tree3.calculate_delay() + accumulator.calculate_delay_msb())
    predicted_delay = max(unitbank.delay + mults.calculate_delay() + adder_tree1.calculate_delay() + adder_tree2.calculate_delay(), adder_tree3.calculate_delay() + accumulator_delay)
        
    ## Energy cost breakdown per cycle
    energy_mults = dimc['input_toggle_rate'] * dimc['banks'] * dimc['output_channel'] * mults.calculate_energy() # fJ
    energy_adder_tree = dimc['input_toggle_rate'] * dimc['weight_sparsity'] * dimc['banks'] * dimc['output_channel'] * ( 8*8*adder_tree1.calculate_energy() + 8*adder_tree2.calculate_energy() + adder_tree3.calculate_energy() ) # fJ
    energy_accumulator = dimc['banks'] * dimc['output_channel'] * accumulator.calculate_energy()
    energy_banks = 0 # make it to zero because: (1) from validation, this cost is very small in percentage to entire macro energy; (2) papaers don't report how many cycles they will read out the data once.
    energy_regs_accumulator = dimc['banks'] * dimc['output_channel'] * regs_accumulator.w_energy
    energy_regs_pipeline = dimc['banks'] * dimc['output_channel'] * 8*regs_pipeline.w_energy
        
    predicted_energy_per_cycle = energy_mults + energy_adder_tree + energy_accumulator + energy_banks + energy_regs_accumulator + energy_regs_pipeline
    
    number_of_cycle = dimc['activation_precision']/dimc['input_precision']
    
    predicted_energy = predicted_energy_per_cycle * number_of_cycle
    
    number_of_operations = 2*dimc['banks']*dimc['output_channel']*dimc['input_channel'] # 1MAC = 2 Operations
    predicted_tops = number_of_operations/(predicted_delay*number_of_cycle) / (10**3)
    predicted_topsw = number_of_operations/predicted_energy * 10**3
    
    ## Energy breakdown per MAC
    number_of_mac = number_of_operations/2
    energy_mults_mac = energy_mults * number_of_cycle/number_of_mac
    energy_adder_tree_mac = energy_adder_tree * number_of_cycle/number_of_mac
    energy_accumulator_mac = energy_accumulator * number_of_cycle/number_of_mac
    energy_banks_mac = energy_banks * number_of_cycle/number_of_mac
    energy_regs_accumulator_mac = energy_regs_accumulator * number_of_cycle/number_of_mac
    energy_regs_pipeline_mac = energy_regs_pipeline * number_of_cycle/number_of_mac
    energy_estimation_per_mac = predicted_energy/number_of_mac
    energy_reported_per_mac = 2000/dimc['TOP/s/W']
    
    area_mismatch = abs(predicted_area/dimc['area']-1)
    delay_mismatch = abs(predicted_delay/dimc['tclk']-1)
    energy_mismatch = abs(energy_estimation_per_mac/energy_reported_per_mac-1)
    return area_mismatch, delay_mismatch, energy_mismatch
    print(area_mults, area_adder_tree, area_accumulator+area_regs_accumulator, area_banks, area_regs_pipeline)
    print(energy_mults_mac, energy_adder_tree_mac, energy_accumulator_mac+energy_regs_accumulator_mac, energy_banks_mac, energy_regs_pipeline_mac)
    pdb.set_trace()
    # return predicted_area, predicted_delay, predicted_energy/number_of_operations