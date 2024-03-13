import pdb
from aimc_cost_model import *
from dimc_cost_model import *

def aimc2_cost_estimation(aimc, cacti_value):
    unit_reg = UnitDff(aimc['unit_area'], aimc['unit_delay'], aimc['unit_cap'])
    unit_area = aimc['unit_area']
    unit_delay = aimc['unit_delay']
    unit_cap = aimc['unit_cap']
    input_channel = aimc['input_channel']
    reg_input_bitwidth = aimc['reg_input_bitwidth']
    input_bandwidth = input_channel * aimc['input_precision']
    output_bandwidth_per_channel = aimc['output_precision']
    """
    multiplier array for each output channel
    """
    col_mux = 2
    mults = MultiplierArray(vdd=aimc['vdd'],input_precision=int(aimc['multiplier_precision']),number_of_multiplier=col_mux*input_channel, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
    
    """
    adder_tree for each output channel
    """
    # adder tree with place value
    adder1 = Adder(vdd=aimc['vdd'], input_precision=7, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap) # 8 in total
    adder2 = Adder(vdd=aimc['vdd'], input_precision=9, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap) # 4 in total
    adder3 = Adder(vdd=aimc['vdd'], input_precision=12, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap) # 2 in total
    adder4 = Adder(vdd=aimc['vdd'], input_precision=15, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap) # 1 in total
    adder_tree = AdderTree(vdd=aimc['vdd'], input_precision=int(aimc['adder_input_precision']), number_of_input=input_channel, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
        
    """
    accumulator for each output channel
    """
    accumulator = Adder(vdd=aimc['vdd'], input_precision=int(aimc['accumulator_precision']), unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
    
    """
    ADC cost for each ADC
    """
    adc = ADC(resolution=aimc['adc_resolution'], ICH=aimc['input_channel'])
    
    """
    DAC cost for each DAC
    """
    dac = DAC(resolution=aimc['dac_resolution'])

    """
    memory instance (delay unit: ns, energy unit: fJ, area unit: mm2)
    unitbank: sram bank, data from CACTI
    regs_accumulator: register files inside accumulator for each output channel (congifuration is same with regs_output)
    """
    unitbank = MemoryInstance(name='unitbank', size=aimc['rows']*aimc['cols'], r_bw=aimc['cols'], w_bw=aimc['cols'], delay=cacti_value['delay']*0, r_energy=cacti_value['r_energy'], w_energy=cacti_value['w_energy'], area=cacti_value['area'], r_port=1, w_port=1, rw_port=0, latency=0)
    regs_accumulator = MemoryInstance(name='regs_accumulator', size=aimc['reg_accumulator_precision'], r_bw=aimc['reg_accumulator_precision'], w_bw=aimc['reg_accumulator_precision'], delay=unit_reg.calculate_delay(), r_energy=0, w_energy=unit_reg.calculate_cap() * aimc['vdd']**2 * aimc['reg_accumulator_precision'], area=unit_reg.calculate_area()*aimc['reg_accumulator_precision'], r_port=1, w_port=1, rw_port=0, latency=0)
    regs_pipeline = MemoryInstance(name='regs_pipeline', size=5*16, r_bw=5*16, w_bw=5*16, delay=0, r_energy=0, w_energy=unit_reg.calculate_cap() * aimc['vdd']**2 * 5 * 16, area=unit_reg.calculate_area()*5*16, r_port=1, w_port=1, rw_port=0, latency=1)
    
    energy_wl = aimc['input_channel'] * aimc['unit_cap']/2*2 * aimc['vdd']**2 * aimc['weight_precision'] # per output channel
    #energy_bl = aimc['rows'] * aimc['unit_cap']/2*2 * aimc['vdd']**2 * aimc['weight_precision'] # per output channel (aimc['unit_cap']/2 for bitline cap/cell, *2 for 2 bitline port of 2 cells connecting together)
    energy_en = aimc['input_channel'] * aimc['unit_cap']/2*2 * aimc['vdd']**2 # per output channel (energy cost on "en" enable signal)
    energy_bl = 0 # assume bitline doesn't change during computation
    
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
    area_mults = aimc['banks'] * aimc['output_channel'] * mults.calculate_area()
    #area_adder_tree = aimc['banks'] * aimc['output_channel'] * adder_tree.calculate_area()
    area_adder_tree = aimc['banks'] * aimc['output_channel'] * ( 8*adder1.calculate_area() + 4*adder2.calculate_area() + 2*adder3.calculate_area() + 1*adder4.calculate_area() )
    area_accumulator = aimc['banks'] * aimc['output_channel'] * accumulator.calculate_area()
    area_banks = aimc['banks'] *unitbank.area
    area_regs_accumulator = aimc['banks'] * aimc['output_channel'] * regs_accumulator.area
    area_regs_pipeline = aimc['banks'] * aimc['output_channel'] * regs_pipeline.area
    area_adc = aimc['banks'] * aimc['output_channel'] * 16 * adc.calculate_area()
    area_dac = aimc['banks'] * 2 * aimc['input_channel'] * dac.calculate_area()
    
    
    predicted_area = area_mults + area_adder_tree + area_accumulator + area_banks + area_regs_accumulator + area_regs_pipeline + area_adc + area_dac# cost of input/output regs has been taken out
    
    ## delay cost (2* for input transfer two times)
    adder_1b_carry_delay = 2*UnitNand2(unit_area, unit_delay, unit_cap).calculate_delay()
    accumulator_delay = accumulator.calculate_delay_lsb()+adder_1b_carry_delay * (aimc['reg_accumulator_precision']-aimc['accumulator_input_precision'])
    predicted_delay = max(2* (unitbank.delay + mults.calculate_delay() + adc.calculate_delay()), 2*(adder_tree.calculate_delay() + accumulator_delay))
        
    ## Energy cost breakdown per input transfer
    energy_mults = (1-aimc['weight_sparsity']) * aimc['banks'] * aimc['output_channel'] * mults.calculate_energy()
    #energy_adder_tree = (1-aimc['weight_sparsity']) * aimc['banks'] * aimc['output_channel'] * adder_tree.calculate_energy()
    energy_adder_tree = (1 - aimc['weight_sparsity']) * aimc['banks'] * aimc[
        'output_channel'] * ( 8*adder1.calculate_energy() + 4*adder2.calculate_energy() + 2*adder3.calculate_energy() + 1*adder4.calculate_energy() )
    energy_accumulator = aimc['banks'] * aimc['output_channel'] * accumulator.calculate_energy()
    energy_banks = (1-aimc['weight_sparsity']) * aimc['banks'] * aimc['output_channel'] * (energy_wl + energy_bl + energy_en)
    energy_regs_accumulator = aimc['banks'] * aimc['output_channel'] * regs_accumulator.w_energy
    energy_regs_pipeline = aimc['banks'] * aimc['output_channel'] * regs_pipeline.w_energy
    energy_adc = (1-aimc['weight_sparsity']) * aimc['banks'] * aimc['output_channel'] * 16 * adc.calculate_energy(vdd=aimc['vdd'])
    energy_dac = aimc['banks'] * 2 * aimc['input_channel'] * dac.calculate_energy(vdd=aimc['vdd'], k0=aimc['dac_energy_k0'])
    
    ## 2* for input transfer two times
    energy_mults *= 2
    energy_adder_tree *= 2
    energy_accumulator *= 2
    energy_banks *= 2
    energy_regs_accumulator *= 2
    energy_regs_pipeline *= 2
    energy_adc *= 2
    energy_dac *= 2
    
        
    predicted_energy_per_cycle = energy_mults + energy_adder_tree + energy_accumulator + energy_banks + energy_regs_accumulator + energy_regs_pipeline + energy_adc + energy_dac
    
    number_of_cycle = aimc['activation_precision']/aimc['input_precision']
    
    predicted_energy = predicted_energy_per_cycle * number_of_cycle
    
    number_of_operations = 2*aimc['banks']*aimc['output_channel']*aimc['input_channel'] # 1MAC = 2 Operations
        
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
    energy_adc_mac = energy_adc * number_of_cycle/number_of_mac
    energy_dac_mac = energy_dac * number_of_cycle/number_of_mac
    energy_estimation_per_mac = predicted_energy/number_of_mac
    energy_reported_per_mac = 2000/aimc['TOP/s/W']
    
    area_mismatch = abs(predicted_area/aimc['area']-1)
    delay_mismatch = abs(predicted_delay/aimc['tclk']-1)
    energy_mismatch = abs(energy_estimation_per_mac/energy_reported_per_mac-1)
    #return predicted_area, predicted_delay, energy_estimation_per_mac
    #return area_mismatch, delay_mismatch, energy_mismatch
    #print(area_mults, area_adder_tree, area_accumulator, area_banks, area_regs_accumulator, area_regs_pipeline)
    #print(energy_mults_mac, energy_adder_tree_mac, energy_accumulator_mac, energy_banks_mac, energy_regs_accumulator_mac, energy_regs_pipeline_mac)
    pdb.set_trace()
