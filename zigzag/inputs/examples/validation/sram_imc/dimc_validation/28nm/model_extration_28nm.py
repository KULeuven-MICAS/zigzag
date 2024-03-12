import pdb
from dimc_validation import *
from dimc_validation4 import *
from dimc_validation_subfunc4 import *

def area_fitting():
    mismatch = 1
    for area in range(294, 1000, 1):
        dimc_ISSCC2022_15_5['unit_area'] = area/1000 #um2
        dimc_ISSCC2022_11_7['unit_area'] = area/1000 #um2
        dimc_ISSCC2023_7_2['unit_area'] = area/1000 #um2
        dimc_ISSCC2023_16_3['unit_area'] = area/1000 #um2
        a1, d1, e1 = dimc_cost_estimation(dimc_ISSCC2022_15_5, cacti_ISSCC2022_15_5)
        a2, d2, e2 = dimc_cost_estimation(dimc_ISSCC2022_11_7, cacti_ISSCC2022_11_7)
        a3, d3, e3 = dimc_cost_estimation(dimc_ISSCC2023_7_2, cacti_value_ISSCC2023_7_2)
        a4, d4, e4 = dimc_cost_estimation4(dimc_ISSCC2023_16_3, cacti_value_ISSCC2023_16_3)
        at = (a1+a2+a3+a4)/4 # average area mismatch
        dt = (d1+d2+d3+d4)/4 # average delay mismatch
        et = (e1+e3)/2 # average energy mismatch (peak energy is not reported in paper2)
        if at < mismatch:
            mismatch = at
            fitted_unit_area = area/1000
    print(f"fitted_unit_area: {fitted_unit_area}, average_mismatch: {mismatch}")
    return mismatch, fitted_unit_area

def delay_fitting():
    mismatch = 1
    for delay in range(150, 500, 1):
        dimc_ISSCC2022_15_5['unit_delay'] = delay/10000 #ns
        dimc_ISSCC2022_11_7['unit_delay'] = delay/10000 #ns
        dimc_ISSCC2023_7_2['unit_delay'] = delay/10000 #ns
        dimc_ISSCC2023_16_3['unit_delay'] = delay/10000 #ns
        a1, d1, e1 = dimc_cost_estimation(dimc_ISSCC2022_15_5, cacti_ISSCC2022_15_5)
        a2, d2, e2 = dimc_cost_estimation(dimc_ISSCC2022_11_7, cacti_ISSCC2022_11_7)
        a3, d3, e3 = dimc_cost_estimation(dimc_ISSCC2023_7_2, cacti_value_ISSCC2023_7_2)
        a4, d4, e4 = dimc_cost_estimation4(dimc_ISSCC2023_16_3, cacti_value_ISSCC2023_16_3)
        at = (a1+a2+a3+a4)/4 # average area mismatch
        dt = (d1+d2+d3+d4)/4 # average delay mismatch
        et = (e1+e3)/2 # average energy mismatch (peak energy is not reported in paper2)
        if dt < mismatch:
            mismatch = dt
            dlist = [d1,d2,d3,d4]
            fitted_unit_delay = delay/10000
    print(f"fitted_unit_delay: {fitted_unit_delay}, average_mismatch: {mismatch}")
    return mismatch, fitted_unit_delay

def cap_fitting():
    mismatch = 1
    for cap in range(1, 50, 1):
        dimc_ISSCC2022_15_5['unit_cap'] = cap/10 #fF
        dimc_ISSCC2022_11_7['unit_cap'] = cap/10 #fF
        dimc_ISSCC2023_7_2['unit_cap'] = cap/10 #fF
        a1, d1, e1 = dimc_cost_estimation(dimc_ISSCC2022_15_5, cacti_ISSCC2022_15_5)
        a2, d2, e2 = dimc_cost_estimation(dimc_ISSCC2022_11_7, cacti_ISSCC2022_11_7)
        a3, d3, e3 = dimc_cost_estimation(dimc_ISSCC2023_7_2, cacti_value_ISSCC2023_7_2)
        at = (a1+a2+a3)/3 # average area mismatch
        dt = (d1+d2+d3)/3 # average delay mismatch
        et = (e1+e3)/2 # average energy mismatch (peak energy is not reported in paper2)
        if et < mismatch:
            mismatch = et
            fitted_unit_cap = cap/10
    print(f"fitted_unit_cap: {fitted_unit_cap}, average_mismatch: {mismatch}")
    return mismatch, fitted_unit_cap

if __name__ == '__main__':
    area_fitting()
    delay_fitting()
    cap_fitting()
