from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance

my_mem_instance = MemoryInstance(
    name="test_dram",
    size=1009001*8,
    r_bw=128,
    w_bw=128,
    auto_cost_extraction=True,
    rw_port=1
)

print(my_mem_instance.r_cost, my_mem_instance.w_cost)