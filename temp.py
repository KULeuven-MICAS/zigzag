from zigzag.api import get_hardware_performance_zigzag_imc

workloads = (
    "inputs/workload/alexnet.onnx",
    "inputs/workload/mobilenetv2.onnx",
    "inputs/workload/resnet18.onnx",
    "inputs/workload/resnet18.yaml",
)

# Expected energy, latency (#cycles), clk time and area for each workload defined above
ens_lats_clks_areas = {
    "inputs/workload/alexnet.onnx": (2557076250.266322, 44012016.0, 6.61184, 0.7892517658006044),
    "inputs/workload/mobilenetv2.onnx": (802185102.578702, 14939020.0, 6.61184, 0.7892517658006044),
    "inputs/workload/resnet18.onnx": (2252151728.145326, 62079022.0, 6.61184, 0.7892517658006044),
    "inputs/workload/resnet18.yaml": (2466090000.2577806, 67309272.0, 6.61184, 0.7892517658006044),
}


def mapping():
    return "inputs/mapping/default_imc.yaml"


def accelerator() -> str:
    return "inputs/hardware/aimc.yaml"


if __name__ == "__main__":
    ans = {}
    for workload in [workloads[0]]:
        accelerator = accelerator()
        mapping = mapping()
        (energy, latency, tclk, area, cmes) = get_hardware_performance_zigzag_imc(workload, accelerator, mapping)
        (expected_energy, expected_latency, expected_tclk, expected_area) = ens_lats_clks_areas[workload]
        ans[workload] = (expected_energy, expected_latency, expected_tclk, expected_area)
