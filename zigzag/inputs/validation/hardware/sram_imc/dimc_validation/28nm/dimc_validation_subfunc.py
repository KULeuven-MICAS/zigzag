import pdb
from dimc_cost_model import *

def dimc_cost_estimation(dimc, cacti_value):
    unit_reg = UnitDff(dimc['unit_area'], dimc['unit_delay'], dimc['unit_cap'])
    unit_area = dimc['unit_area']
    unit_delay = dimc['unit_delay']
    unit_cap = dimc['unit_cap']
    input_channel = dimc['input_channel']
    reg_input_bitwidth = dimc['reg_input_bitwidth']
    input_bandwidth = input_channel * dimc['input_precision']
    output_bandwidth_per_channel = dimc['output_precision']
    """
    multiplier array for each output channel
    """
    if dimc['booth_encoding'] == True:
        mults = MultiplierArray(vdd=dimc['vdd'],input_precision=int(dimc['multiplier_precision']),number_of_multiplier=input_channel*dimc['input_precision']/2, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
    else:
        mults = MultiplierArray(vdd=dimc['vdd'],input_precision=int(dimc['multiplier_precision']),number_of_multiplier=input_channel*dimc['input_precision'], unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
    
    """
    adder_tree for each output channel
    """
    adder_tree = AdderTree(vdd=dimc['vdd'], input_precision=int(dimc['adder_input_precision']), number_of_input=input_channel, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
        
    """
    accumulator for each output channel
    """
    accumulator = Adder(vdd=dimc['vdd'], input_precision=int(dimc['accumulator_precision']), unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)

    """
    memory instance (delay unit: ns, energy unit: fJ, area unit: mm2)
    unitbank: sram bank, data from CACTI
    regs_input: input register files
    regs_output: output register files for each output channel
    regs_accumulator: register files inside accumulator for each output channel (congifuration is same with regs_output)
    """
    # bank delay is neglected for delay validation (due to small contribution; and RBL delay is also included, therefore discrepancy exists.
    unitbank = MemoryInstance(name='unitbank', size=dimc['rows']*dimc['cols'], r_bw=dimc['cols'], w_bw=dimc['cols'], delay=0, r_energy=cacti_value['r_energy'], w_energy=cacti_value['w_energy'], area=cacti_value['area'], r_port=1, w_port=1, rw_port=0, latency=0)
    regs_input = MemoryInstance(name='regs_input', size=reg_input_bitwidth, r_bw=reg_input_bitwidth, w_bw=reg_input_bitwidth, delay=unit_reg.calculate_delay(), r_energy=0, w_energy=unit_reg.calculate_cap() * dimc['vdd']**2 * reg_input_bitwidth, area=unit_reg.calculate_area()*reg_input_bitwidth, r_port=1, w_port=1, rw_port=0, latency=1)
    regs_output = MemoryInstance(name='regs_output',size=output_bandwidth_per_channel, r_bw=output_bandwidth_per_channel, w_bw=output_bandwidth_per_channel, delay=unit_reg.calculate_delay(), r_energy=0, w_energy=unit_reg.calculate_cap() * dimc['vdd']**2 * output_bandwidth_per_channel, area=unit_reg.calculate_area()*output_bandwidth_per_channel, r_port=1, w_port=1, rw_port=0, latency=1)
    regs_accumulator = MemoryInstance(name='regs_accumulator', size=dimc['reg_accumulator_precision'], r_bw=dimc['reg_accumulator_precision'], w_bw=dimc['reg_accumulator_precision'], delay=unit_reg.calculate_delay(), r_energy=0, w_energy=unit_reg.calculate_cap() * dimc['vdd']**2 * dimc['reg_accumulator_precision'], area=unit_reg.calculate_area()*dimc['reg_accumulator_precision'], r_port=1, w_port=1, rw_port=0, latency=0)
    # pipeline after adder tree and before accumulator
    if dimc['pipeline'] == True:
        pipeline_bw_per_channel = dimc['reg_pipeline_precision']
        regs_pipeline = MemoryInstance(name='regs_pipeline', size=pipeline_bw_per_channel, r_bw=pipeline_bw_per_channel, w_bw=pipeline_bw_per_channel, delay=unit_reg.calculate_delay(), r_energy=0, w_energy=unit_reg.calculate_cap() * dimc['vdd']**2 * pipeline_bw_per_channel, area=unit_reg.calculate_area()*pipeline_bw_per_channel, r_port=1, w_port=1, rw_port=0, latency=1)
    else:
        regs_pipeline = MemoryInstance(name='regs_pipeline', size=0, r_bw=0, w_bw=0, delay=0, r_energy=0, w_energy=0, area=0, r_port=1, w_port=1, rw_port=0, latency=0)
    
    ################### special cost for each paper ##################################
    """
    special cost for ISSCC2023, 7.2: adder tree across output channels
    """
    if dimc['paper_idx'] == 'ISSCC2023, 7.2':
        adder_tree_channel = AdderTree(vdd=dimc['vdd'], input_precision=16, number_of_input=8, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
        
    ##################################################################################
    
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
    area_adder_tree = dimc['banks'] * dimc['output_channel'] * adder_tree.calculate_area()
    area_accumulator = dimc['banks'] * dimc['output_channel'] * accumulator.calculate_area()
    area_banks = dimc['banks'] * unitbank.area
    area_regs_input = dimc['banks'] * regs_input.area
    area_regs_output = dimc['banks'] * dimc['output_channel'] * regs_output.area
    area_regs_accumulator = dimc['banks'] * dimc['output_channel'] * regs_accumulator.area
    area_regs_pipeline = dimc['banks'] * dimc['output_channel'] * regs_pipeline.area
    
    if dimc['paper_idx'] == 'ISSCC2022, 15.5': # extra area cost for supporting FP operation
        extra_adder_tree = AdderTree(vdd=dimc['vdd'], input_precision=32, number_of_input=2, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
        extra_accumulator = Adder(vdd=dimc['vdd'], input_precision=32, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
        extra_regs_accumulator = MemoryInstance(name='extra_regs_accumulator', size=32, r_bw=32, w_bw=32, delay=0, r_energy=0, w_energy=0, area=1.764/(10**6)*32, r_port=1, w_port=1, rw_port=0, latency=0)
        area_extra_adder_tree = dimc['banks'] * 5 * extra_adder_tree.calculate_area()
        area_extra_accumulator = dimc['banks'] * 5 * extra_accumulator.calculate_area()
        area_extra_regs_accumulator = dimc['banks'] * 5 * extra_regs_accumulator.area
        
        area_adder_tree += area_extra_adder_tree
        area_accumulator += area_extra_accumulator
        area_regs_accumulator += area_extra_regs_accumulator
        
    if dimc['paper_idx'] == 'ISSCC2022, 11.7':
        area_accumulator = dimc['banks'] * dimc['output_channel'] * dimc['input_channel'] * accumulator.calculate_area()
        area_regs_accumulator = dimc['banks'] * dimc['output_channel'] * dimc['input_channel'] * regs_accumulator.area
        area_regs_input = regs_input.area # input regs are shared across banks
        area_regs_pipeline = dimc['banks'] * dimc['output_channel'] * dimc['input_channel'] * regs_pipeline.area
    if dimc['paper_idx'] == 'ISSCC2023, 7.2':
        area_adder_tree_channel = dimc['banks'] * adder_tree_channel.calculate_area()
        area_adder_tree += area_adder_tree_channel
        area_accumulator = dimc['banks'] * accumulator.calculate_area()
        area_regs_output = dimc['banks'] * regs_output.area
        area_regs_accumulator = dimc['banks'] * regs_accumulator.area
    
    predicted_area = area_mults + area_adder_tree + area_accumulator + area_banks + area_regs_input * 0 + area_regs_output * 0 + area_regs_accumulator + area_regs_pipeline # cost of input/output regs has been taken out
    
    ## Minimum clock time 
    adder_1b_carry_delay = 2*UnitNand2(unit_area, unit_delay, unit_cap).calculate_delay()
    accumulator_delay = accumulator.calculate_delay_lsb()+adder_1b_carry_delay * (dimc['reg_accumulator_precision']-dimc['accumulator_input_precision'])
    if dimc['pipeline'] == True:
        if dimc['paper_idx'] == 'ISSCC2022, 11.7':  # for dimc2
            #predicted_delay = max(unitbank.delay + mults.calculate_delay(), adder_tree.calculate_delay() + accumulator.calculate_delay_msb())
            accumulator_delay = accumulator.calculate_delay_lsb()
            predicted_delay = max(unitbank.delay + mults.calculate_delay(), adder_tree.calculate_delay() + accumulator_delay)
        else:
            #predicted_delay = max(unitbank.delay + mults.calculate_delay() + adder_tree.calculate_delay(), accumulator.calculate_delay_msb())
            predicted_delay = max(unitbank.delay + mults.calculate_delay() + adder_tree.calculate_delay(), accumulator_delay)
    else:
        if dimc['paper_idx'] == 'ISSCC2023, 7.2':  # for dimc3
            #predicted_delay = unitbank.delay + mults.calculate_delay() + adder_tree.calculate_delay() + accumulator.calculate_delay_msb() + adder_tree_channel.calculate_delay()
            predicted_delay = unitbank.delay + mults.calculate_delay() + adder_tree.calculate_delay() + accumulator_delay + adder_tree_channel.calculate_delay()
        else:  # for dimc1
            #predicted_delay = unitbank.delay + mults.calculate_delay() + adder_tree.calculate_delay() + accumulator.calculate_delay_msb()
            predicted_delay = unitbank.delay + mults.calculate_delay() + adder_tree.calculate_delay() + accumulator_delay
        
    ## Energy cost breakdown per cycle
    energy_mults = dimc['input_toggle_rate'] * dimc['weight_sparsity'] * dimc['banks'] * dimc['output_channel'] * mults.calculate_energy()
    energy_adder_tree = dimc['input_toggle_rate'] * dimc['weight_sparsity'] * dimc['banks'] * dimc['output_channel'] * adder_tree.calculate_energy()
    energy_accumulator = dimc['banks'] * dimc['output_channel'] * accumulator.calculate_energy()
    energy_banks = dimc['banks'] * unitbank.r_energy * 0 # make it to zero because: (1) from validation, this cost is very small in percentage to entire macro energy; (2) papaers don't report how many cycles they will read out the data once.
    energy_regs_input = dimc['banks'] * regs_input.w_energy
    energy_regs_output = dimc['banks'] * dimc['output_channel'] * regs_output.w_energy
    energy_regs_accumulator = dimc['banks'] * dimc['output_channel'] * regs_accumulator.w_energy
    energy_regs_pipeline = dimc['banks'] * dimc['output_channel'] * regs_pipeline.w_energy
    
    if dimc['paper_idx'] == 'ISSCC2022, 11.7':
        pass
    if dimc['paper_idx'] == 'ISSCC2023, 7.2':
        energy_adder_tree_channel = dimc['banks'] * adder_tree_channel.calculate_energy()
        energy_adder_tree += energy_adder_tree_channel
        energy_accumulator = dimc['banks'] * accumulator.calculate_energy()
        energy_regs_output = dimc['banks'] * regs_output.w_energy
        energy_regs_accumulator = dimc['banks'] * regs_accumulator.w_energy
        
    predicted_energy_per_cycle = energy_mults + energy_adder_tree + energy_accumulator + energy_banks + energy_regs_accumulator + energy_regs_pipeline # + energy_regs_input + energy_regs_output
    
    number_of_cycle = dimc['activation_precision']/dimc['input_precision']
    
    predicted_energy = predicted_energy_per_cycle * number_of_cycle
    
    number_of_operations = 2*dimc['banks']*dimc['output_channel']*dimc['input_channel'] # 1MAC = 2 Operations
    if dimc['paper_idx'] == 'ISSCC2023, 7.2':
        number_of_operations = 2*dimc['banks']*dimc['output_channel']*dimc['input_channel']/dimc['weight_precision'] # 1MAC = 2 Operations
        
    predicted_tops = number_of_operations/(predicted_delay*number_of_cycle) / (10**3)
    predicted_topsw = number_of_operations/predicted_energy * 10**3
    
    ## Energy breakdown per MAC
    number_of_mac = number_of_operations/2
    energy_mults_mac = energy_mults * number_of_cycle/number_of_mac
    energy_adder_tree_mac = energy_adder_tree * number_of_cycle/number_of_mac
    energy_accumulator_mac = energy_accumulator * number_of_cycle/number_of_mac
    energy_banks_mac = energy_banks * number_of_cycle/number_of_mac
    # energy_regs_input_mac = energy_regs_input * number_of_cycle/number_of_mac
    # energy_regs_output_mac = energy_regs_output * number_of_cycle/number_of_mac
    energy_regs_accumulator_mac = energy_regs_accumulator * number_of_cycle/number_of_mac
    energy_regs_pipeline_mac = energy_regs_pipeline * number_of_cycle/number_of_mac
    energy_estimation_per_mac = predicted_energy/number_of_mac
    energy_reported_per_mac = 2000/dimc['TOP/s/W']
    
    area_mismatch = abs(predicted_area/dimc['area']-1)
    delay_mismatch = abs(predicted_delay/dimc['tclk']-1)
    energy_mismatch = abs(energy_estimation_per_mac/energy_reported_per_mac-1)
    return area_mismatch, delay_mismatch, energy_mismatch
    #print(area_mults, area_adder_tree, area_accumulator+area_regs_accumulator, area_banks, area_regs_pipeline)
    #print(energy_mults_mac, energy_adder_tree_mac, energy_accumulator_mac+energy_regs_accumulator_mac, energy_banks_mac, energy_regs_pipeline_mac)
    pdb.set_trace()
    # return predicted_area, predicted_delay, predicted_energy/number_of_operations