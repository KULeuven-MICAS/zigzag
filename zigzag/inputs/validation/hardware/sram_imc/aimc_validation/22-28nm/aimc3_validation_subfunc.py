import pdb
from aimc_cost_model import *
from dimc_cost_model import *

def aimc3_cost_estimation(aimc, cacti_value):
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
    mults = MultiplierArray(vdd=aimc['vdd'],input_precision=int(aimc['multiplier_precision']),number_of_multiplier=64, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
    mults_energy = MultiplierArray(vdd=aimc['vdd'],input_precision=int(aimc['multiplier_precision']),number_of_multiplier=16, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap) # mapped multipliers
    
    """
    adder_tree for each output channel
    """
    adder_tree = Adder(vdd=aimc['vdd'], input_precision=6, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
        
    """
    accumulator for each output channel
    """
    accumulator = None
    
    """
    ADC cost for each ADC
    """
    adc = ADC(resolution=aimc['adc_resolution'], ICH=aimc['input_channel'])
    adc_area = ADC(resolution=7, ICH=64) # for estimating area
    
    """
    DAC cost for each DAC
    """
    dac = None

    """
    memory instance (delay unit: ns, energy unit: fJ, area unit: mm2)
    unitbank: sram bank, data from CACTI
    regs_accumulator: register files inside accumulator for each output channel (congifuration is same with regs_output)
    """
    unitbank = MemoryInstance(name='unitbank', size=aimc['rows']*aimc['cols'], r_bw=aimc['cols'], w_bw=aimc['cols'], delay=cacti_value['delay']*0, r_energy=cacti_value['r_energy'], w_energy=cacti_value['w_energy'], area=cacti_value['area'], r_port=1, w_port=1, rw_port=0, latency=0)
    regs_accumulator = None
    regs_pipeline = None
    
    energy_wl = 0 # per output channel
    # for energy_bl, there are 64 rows in total, but only 8 are used.
    energy_bl = aimc['input_channel'] * aimc['unit_cap']/2 * aimc['vdd']**2 * aimc['weight_precision'] # per output channel (aimc['unit_cap']/2 for bitline cap/cell, *2 for 2 bitline port of 2 cells connecting together)
    energy_en = 0 # per output channel (no enable signal) (for cap couplling-based AIMC, no enable signal is required, as long as the input sequence timing is gated when not under computation.)
    
    
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
    area_adder_tree = aimc['banks'] * aimc['output_channel'] * adder_tree.calculate_area()
    area_accumulator = 0
    if aimc['compact_rule'] == False:
        area_banks = aimc['banks'] * 3*unitbank.area # 2 for non-compact rule scaling
    else:
        area_banks = aimc['banks'] *unitbank.area
    area_regs_accumulator = 0
    area_regs_pipeline = 0
    area_adc = aimc['banks'] * aimc['output_channel'] * adc_area.calculate_area()
    area_dac = 0 # =0
    
    # (for beyong ADC/DAC part, scale from 28nm -> 22nm, exclude ADC/DAC, which is assumed indepedent from tech.) (assume linear)
    area_mults = area_mults/28*22
    area_adder_tree = area_adder_tree/28*22
    area_accumulator = area_accumulator/28*22
    area_banks = area_banks/28*22
    area_regs_accumulator = area_regs_accumulator/28*22
    area_regs_pipeline = area_regs_pipeline/28*22
    area_adc = area_adc/28*22
    area_dac = area_dac/28*22
    
    
    predicted_area = area_mults + area_adder_tree + area_accumulator + area_banks + area_regs_accumulator + area_regs_pipeline + area_adc + area_dac# cost of input/output regs has been taken out
    
    ## delay cost
    predicted_delay = unitbank.delay + mults.calculate_delay() + adder_tree.calculate_delay_msb() + adc.calculate_delay()
        
    ## Energy cost breakdown per input transfer
    energy_mults = aimc['input_toggle_rate'] * (1-aimc['weight_sparsity']) * aimc['banks'] * aimc['output_channel'] * mults_energy.calculate_energy()
    energy_adder_tree = (1-aimc['weight_sparsity']) * aimc['banks'] * aimc['output_channel'] * adder_tree.calculate_energy()
    energy_accumulator = 0
    energy_banks = (1-aimc['weight_sparsity']) * aimc['banks'] * aimc['output_channel'] * (energy_wl + energy_bl + energy_en)
    energy_regs_accumulator = 0
    energy_regs_pipeline = 0
    energy_adc = (1-aimc['weight_sparsity']) * aimc['banks'] * aimc['output_channel'] * adc.calculate_energy(vdd=aimc['vdd'])
    energy_dac = 0
    
    # (for beyong ADC/DAC part, scale from 28nm -> 22nm, exclude ADC/DAC, which is assumed indepedent from tech.) (assume linear)
    energy_mults = energy_mults/28*22
    energy_adder_tree = energy_adder_tree/28*22
    energy_accumulator = energy_accumulator/28*22
    energy_banks = energy_banks/28*22
    energy_regs_accumulator = energy_regs_accumulator/28*22
    energy_regs_pipeline = energy_regs_pipeline/28*22
    
        
    predicted_energy_per_cycle = energy_mults + energy_adder_tree + energy_accumulator + energy_banks + energy_regs_accumulator + energy_regs_pipeline + energy_adc + energy_dac
    
    number_of_cycle = aimc['activation_precision']/aimc['input_precision']
    
    predicted_energy = predicted_energy_per_cycle * number_of_cycle
    
    number_of_operations = 2*aimc['banks']*aimc['output_channel']*aimc['rows']/8 # 1MAC = 2 Operations
        
    predicted_tops = number_of_operations/(predicted_delay*number_of_cycle) / (10**3)
    predicted_topsw = number_of_operations/predicted_energy * 10**3
    
    ## Energy breakdown per MAC
    number_of_mac = number_of_operations/2
    energy_mults_mac = energy_mults * number_of_cycle/number_of_mac
    energy_adder_tree_mac = energy_adder_tree * number_of_cycle/number_of_mac
    energy_accumulator_mac = energy_accumulator * number_of_cycle/number_of_mac
    energy_banks_mac = energy_banks * number_of_cycle/number_of_mac
    energy_regs_accumulator_mac = energy_regs_accumulator * number_of_cycle/number_of_mac
    energy_regs_pipeline_mac = 0
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
