from typeguard import typechecked
from zigzag.datatypes import LayerOperand


@typechecked
class PortActivity:
    """!   Class that collects all the data transfer rate (periodic) information for each DTL (data transfer link)."""

    def __init__(
        self,
        real_cycle: int,
        allowed_cycle: int,
        period: int,
        period_count: int,
        layer_op: LayerOperand,
        mem_lv: int,
        mov_dir: str,
    ):
        """!  The class constructor
        @param real_cycle Within each period, the actual number of cycles used for transferring the amount of data, depended on the memory bw and the data amount to be transferred at that memory level.
        @param period The turnaround cycle at that memory level, which equals to the product of all the temporal loops of current and below memory level.
        @param period_count The total number of period across the whole NN layer computation.

        """
        ## Within each period, the actual number of cycles used for transferring the amount of data, depended on the memory bw and the data amount to be transferred at that memory level.
        self.real_cycle = real_cycle
        self.allowed_cycle = allowed_cycle
        ## The turnaround cycle at that memory level, which equals to the product of all the temporal loops of current and below memory level.
        self.period = period
        ## The total number of period across the whole NN layer computation.
        self.period_count = period_count
        self.served_op_lv_dir = (layer_op, mem_lv, mov_dir)
        """ stalling (+) or slacking (-) cycle in one period """
        self.SS_per_period = real_cycle - allowed_cycle
        """ stalling (+) or slacking (-) cycle in total computation """
        self.SS = (real_cycle - allowed_cycle) * (period_count - 1)
        """ total memory updating window allowed """
        self.MUW = allowed_cycle * (period_count - 1)

    def __str__(self):
        return str(self.served_op_lv_dir)

    def __repr__(self):
        return str(self.served_op_lv_dir)

    def __eq__(self, other) -> bool:
        return str(self.served_op_lv_dir) == other

    def __hash__(self):
        return hash(self.served_op_lv_dir)


@typechecked
class PortBeginOrEndActivity:
    """!  Class that collects all the data transfer rate information for each DTL (data transfer link)."""

    def __init__(
        self,
        real_cycle: int,
        data_in_charge: int,
        mem_bw: int,
        layer_op: LayerOperand,
        mem_lv: int,
        mov_dir: str,
    ):
        """!  The class constructor
        @param real_cycle The actual number of cycles used for transferring the amount of data, depended on the memory
          bw and the data amount to be transferred at that memory level
        @param data_in_charge One-period data transfer amount (bit)
        @param mem_bw Unit: bit/cycle

        """
        ## the actual number of cycles used for transferring the amount of data, depended on the memory bw and the data amount to be transferred at that memory level
        self.real_cycle = real_cycle
        ## one-period data transfer amount (bit)
        self.data_in_charge = data_in_charge
        ## bit/cycle
        self.mem_bw = mem_bw
        self.served_op_lv_dir = (layer_op, mem_lv, mov_dir)

    def __str__(self):
        return str(self.served_op_lv_dir)

    def __repr__(self):
        return str(self.served_op_lv_dir)
