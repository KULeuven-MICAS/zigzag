from math import gcd, prod
import re
from collections import defaultdict
from typing import Dict, List
from copy import deepcopy
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from collections import defaultdict

class LayerNode:

    def __init__(self, layer_id, layer_attrs, node_name=""):
        """
        To construct each layer node, algorithm equation/dimension/indirect relation are parsed.
        This parser collects information of operand, loop dimension, and loop relevance.
        Equal-to-1 loop dimensions are eliminated.

        :param layer_id: The identifier (key) of the layer, as defined in the workload
        :param layer_attrs: contains attributes specified below:
        :param node_name: an optional name for the Node. E.g. the node's name from the onnx model.
        *equation: core computation equation, e.g. 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]',
        'Y[i][j] += A[i][k] * B[k][j]', 'Y[i][j] += A[i][k][l] * B[k][j] * C[l][j]', etc.
        *loop_dim_size: size of each computation loop, e.g. {'B': 1, 'K': 32, 'C': 64, 'OY': 28, 'OX': 28,
        'FY': 1, 'FX': 1, 'G': 1}.
        *equation_relations: for the operand dimension that is not directly a loop dimension,
        a set of specific relation equations between them (operand dimension and loop dimension) is required,
        e.g. ['ix=ox+fx-1', 'iy=oy+fy-1'].
        *core_allocation: the accelerator core on which this layer is executed
        *memory_operand_links: the link between layer operands and where they are stored in the memory hierarchy.

        :return (self)
        ------- directly get from inputs -------
        - loop_dim_size: collection of loop dimension size that >1.
        - operand_precision
        - loop_dim_list, e.g. ['B', 'K', 'C', ...], collection of loop dimension whose size >1.
        - operand_list, e.g. ['W', 'I', 'O']

        ------- operand and loop dimension relation -------
        - operand_loop_dim: operand and loop dimension relationship, e.g.
        operand_loop_dim = {'O': {'r': ['B', 'K', 'OY', 'OX'], 'ir': ['C', 'FX', 'FY'], 'pr': {}},
                            'W': {'r': ['K', 'C', 'FY', 'FX'], 'ir': ['B', 'OX', 'OY'], 'pr': {}},
                            'I': {'r': ['B', 'C'], 'ir': ['K'], 'pr': {'IY': ('OY', 'FY'), 'IX': ('OX', 'FX')}}}

        ------- basic layer information extraction -------
        - total_MAC_count
        - operand_size_elem
        - operand_size_bit
        - operand_data_reuse
        """

        self.id = layer_id
        self.layer_attrs = layer_attrs
        self.name = node_name

        # equation_relations has been replaced by dimension_relations.
        # Check if layer has equation_relations and notify user.
        if 'equation_relations' in layer_attrs:
            raise ValueError(f"Please replace equation_relations by dimension_relations for layer {self}.")

        '''Get required attributes from layer_attrs'''
        equation: str = layer_attrs.get('equation')
        loop_dim_size: Dict[str, int] = layer_attrs.get('loop_dim_size')
        pr_loop_dim_size: Dict[str, int] = layer_attrs.get('pr_loop_dim_size', None)
        operand_precision: Dict[str, int] = layer_attrs.get('operand_precision')
        dimension_relations: List[str] = layer_attrs.get('dimension_relations', [])
        user_spatial_mapping: Dict[str, tuple] = layer_attrs.get('spatial_mapping', None)
        user_temporal_ordering = layer_attrs.get('temporal_ordering', None)
        core_allocation: int = layer_attrs.get('core_allocation', None)
        memory_operand_links: Dict[str, str] = layer_attrs.get('memory_operand_links', None)
        source_storage_level: int = layer_attrs.get('source_storage_level', {})
        operand_source_dimension_mapping: Dict[Dict[str, str]] = layer_attrs.get('operand_source_dimension_mapping', {})
        constant_operands: List[str] = layer_attrs.get('constant_operands', [])
        input_operand_source: Dict[str, list] = layer_attrs.get('operand_source', dict())
        # Save the padding for different tensor dimensions
        padding: Dict[str, tuple] = layer_attrs.get('padding', {})  # Empty dict signals no padding in any dimension

        self.equation = equation
        self.loop_dim_size = dict(item for item in tuple(loop_dim_size.items()))  # if item[1] != 1)
        self.pr_loop_dim_size = pr_loop_dim_size
        self.operand_precision = operand_precision
        self.dimension_relations = dimension_relations
        self.loop_dim_list = list(loop_dim_size.keys())
        self.user_spatial_mapping = user_spatial_mapping
        self.user_temporal_ordering = user_temporal_ordering
        self.core_allocation = core_allocation
        self.memory_operand_links = memory_operand_links.copy()
        self.source_storage_level = source_storage_level
        self.operand_source_dimension_mapping = operand_source_dimension_mapping
        self.constant_operands = constant_operands
        self.padding = padding

        ''' Step1: extract partially-relevant data dimension and its relation to loop dimensions. '''
        pr_loop, pr_loop_list, pr_scaling_factors = self.build_pr_funcs()
        self.pr_loop = pr_loop
        self.pr_scaling_factors = pr_scaling_factors
        if not self.pr_loop_dim_size:
            self.pr_loop_dim_size = {dim: self.calc_pr_dimension_size_total(dim) for dim in pr_loop}

        ''' Step2: extract relevant and irrelevant loop dimensions. '''
        operand_loop_dim, operand_loop_dim_reform, operand_list, operand_dimensionality_order = \
            self.extract_r_ir_loop_info(equation, loop_dim_size, pr_loop, pr_loop_list)
        self.operand_loop_dim = operand_loop_dim
        self.operand_loop_dim_reform = operand_loop_dim_reform
        self.output_operand = operand_list[0]
        self.input_operands = operand_list[1:]
        self.operand_list = operand_list
        self.input_operand_source = input_operand_source
        self.operand_dimensionality_order = operand_dimensionality_order

        # Save the variable (non-constant) input operands
        self.variable_input_operands: list = [op for op in self.input_operands if op not in self.constant_operands]
        # Save the way an operand's tensor should be reshaped for interaction with other nodes.
        self.operand_tensor_reshape: Dict[str, list] = layer_attrs.get('operand_tensor_reshape',  {op: [] for op in self.operand_list})

        ''' Step3: extract layer info, e.g. total operand size, total operand data reuse, total MAC operation, etc. '''
        self.extract_layer_info()


    def build_pr_funcs(self):
        # 1 long dimensions are removed in self.loop_dim_size but required in extract_pr_loop_info
        loop_dim_size = defaultdict(lambda: 1)
        loop_dim_size.update(self.loop_dim_size)
        if self.dimension_relations:
            pr_loop, pr_loop_list, pr_scaling_factors = self.extract_pr_loop_info(self.dimension_relations)
        else:
            pr_loop, pr_loop_list, pr_scaling_factors = {}, [], {}

        return pr_loop, pr_loop_list, pr_scaling_factors

    def get_core_allocation(self):
        return self.core_allocation

    def __str__(self):
        return f"LayerNode_{self.id}"

    def __repr__(self):
        return str(self)
    
    def __jsonrepr__(self):
        """
        JSON representation used for saving this object to a json file.
        """
        return {"equation": self.equation,
                "equation_relations": self.dimension_relations,
                "loop_dimensions": self.loop_dim_size,
                "operand_precision": self.operand_precision,
                "core_allocation": self.core_allocation,
                "user_spatial_mapping": self.user_spatial_mapping,
                "memory_operand_links": self.memory_operand_links,
                "source_storage_level": self.source_storage_level}

    def calc_tensor_size(self, layer_op, loop_sizes):
        """
        Calculates the tensor size (nb of elements) for the given operand layer_op with the given loop dimension sizes loop_sizes.
        :param layer_op: str. A String representing the layer operand for which to compute the tensor size.
        :param loop_sizes: dict. A dict with string keys representing the dimension and integer values representing the size.
        """
        return prod(self.calc_tensor_dims(layer_op, loop_sizes).values())
        # Initialize the tensor size as 1

    def calc_tensor_dim(self, loop_sizes, dim):
        if dim in loop_sizes:
            return loop_sizes[dim]
        elif dim in self.pr_loop:
            related_dimension_sizes = [loop_sizes[dimension] for dimension in self.pr_loop[dim]]
            scaling_factors = list(self.pr_scaling_factors[dim].values())
            assert len(related_dimension_sizes) == len(scaling_factors) == 2, "Shouldn't happen if partial relevancy checks in extract_pr_loop_info() are done correctly."
            args = (val for pair in zip(scaling_factors, related_dimension_sizes) for val in pair)
            pr_dim_size = self.calc_pr_dimension_size(*args)
            # Clip this to the largest possible size for this partially relevant dimension (computed at initialization based on padding)
            pr_dim_size = min(self.pr_loop_dim_size[dim], pr_dim_size)
            return pr_dim_size
        elif dim in self.loop_dim_size:
            assert self.loop_dim_size[dim] == 1, "This line should only be reached when the dim has a size of 1 in the layer."
            return 1
        else:
            raise ValueError("Something went wrong in the initialization of the layer, or in the caller function.")

    def calc_tensor_dims(self, layer_op, loop_sizes):
        out = {}
        op_dimensions = self.operand_loop_dim[layer_op]
        for dim in op_dimensions['r'] + list(op_dimensions['pr'].keys()):
            out[dim] = self.calc_tensor_dim(loop_sizes, dim)
        return out

    def calc_pr_dimension_size_total(self, dim):
        """Compute the total pr dimension size of this node, taking padding into account.

        Args:
            dim (str): The partially relevant dimension, e.g. 'IX'.

        Returns:
            int: The total partially relevant dimension size
        """
        related_dimension_sizes = [self.loop_dim_size[related_dim] for related_dim in self.pr_loop[dim]]
        scaling_factors = list(self.pr_scaling_factors[dim].values())  # assumes this dict is ordered
        assert len(related_dimension_sizes) == len(scaling_factors) == 2, "Shouldn't happen if partial relevancy checks in extract_pr_loop_info() are done correctly."
        args = (val for pair in zip(scaling_factors, related_dimension_sizes) for val in pair)
        total_pr_dim_size = self.calc_pr_dimension_size(*args)
        # Partially relevant loop dimensions can also have padding, so get the padding for this pr dimension and subtract
        padding = self.padding.get(dim, (0, 0))  # default = (0, 0)
        total_pr_dim_size_without_padding = int(total_pr_dim_size - sum(padding))
        return total_pr_dim_size_without_padding


    @staticmethod
    def calc_pr_dimension_size(sa, A, sb, B):
        """
        Calculates the number of unique indices c generated by iterating through the indices
        a in range(0,A,1) and b in range(0,B,1) according to the equation c = sa * a + sb * b.
        sa and sb thus represent the scaling of a, resp. b.
        """
        return int(A*B - max(0, B - (sa/gcd(sa,sb))) * (A - (sb/gcd(sa,sb))))


    @staticmethod
    def return_lambda(equal_sign_right):
        return eval("lambda n: " + equal_sign_right)

    def extract_pr_loop_info(self, equation_relations):
        pr_loop: Dict[str, list] = {}
        pr_loop_list: List[str] = []
        pr_scaling_factors: Dict[str, list] = {}
        padding: Dict[str, int] = {}
        for relation in equation_relations:
            relation_disassembly = re.findall('[a-zA-Z]+', relation)
            
            assert len(relation_disassembly) == 3, f"equation_relation {relation} does not involve a linear relationship between two dimension iterators."
            
            key = relation_disassembly[0].upper()
            val = [loop_dim.upper() for loop_dim in relation_disassembly[1:]]
            pr_loop[key] = val
            pr_loop_list.extend([key] + val)

            # To extract the scaling factors for the different loop dimension iterators, we need to make sure
            # there is a scaling factor present in the equation. If it is not present, raise an exception.
            scaling_factors = {}
            for val_lower in relation_disassembly[1:]:
                if relation[relation.index(val_lower)-1] == '*':
                    if not relation[relation.index(val_lower)-2].isdigit():
                        raise NotImplementedError(f"Please use a scaling factor for every dimension iterator on the RHS of equation {relation}")
                    else:
                        scaling_factors[val_lower] = int(re.findall('(\\d+)(?=\\*'+val_lower+')', relation)[0])
                else:
                    scaling_factors[val_lower] = 1
            #scaling_factors = re.findall('[0-9]+', relation)
            assert len(scaling_factors) == 2, f"Please remove any constants in the equation relation {relation}."
            pr_scaling_factors[key] = scaling_factors

        return pr_loop, pr_loop_list, pr_scaling_factors

    @staticmethod
    def extract_r_ir_loop_info(equation, loop_dim_size, pr_loop, pr_loop_list):
        operand_loop_dim: Dict[str, Dict] = {}
        operand_list = []
        equation = equation.replace('*', ' * ')
        equation = equation.replace('=', ' = ')
        equation = equation.replace('+', ' + ')
        equation_disassembly = re.findall('[a-zA-Z,0-9,=,*,+]+', equation)
        # filter out + that directly precedes an = (+=) or another + (++) to make this work for concat and add
        prev_char = None
        for i, char in enumerate(equation_disassembly):
            if (char == '=' or char == '+') and prev_char == '+':
                equation_disassembly.pop(i-1)
            prev_char = char
        split_location = [i for (i, x) in enumerate(equation_disassembly) if x in ['=', '*', '+']] + [
            len(equation_disassembly)]
        dimension_list = list(loop_dim_size.keys())
        begin_idx = 0
        operand_dimensionality_order = {}
        for split_loc in split_location:
            operand = equation_disassembly[begin_idx]
            operand_list.append(operand)
            operand_loop_dim[operand] = {}
            r_loop_list = [loop_dim.upper() for loop_dim in equation_disassembly[begin_idx + 1:split_loc]]
            ir_loop_list = list(set(dimension_list).difference(r_loop_list))

            pr_loop_remove_flag = any(loop in list(pr_loop.keys()) for loop in r_loop_list)
            if pr_loop_remove_flag:
                operand_loop_dim[operand]['r'] = [loop for loop in r_loop_list if
                                                  loop not in pr_loop_list] #  and loop_dim_size[loop] != 1]
                operand_loop_dim[operand]['ir'] = [loop for loop in ir_loop_list if
                                                   loop not in pr_loop_list and loop_dim_size[loop] != 1]
                operand_loop_dim[operand]['pr'] = pr_loop
            else:
                operand_loop_dim[operand]['r'] = [loop for loop in r_loop_list if loop_dim_size[loop] != 1]
                operand_loop_dim[operand]['ir'] = [loop for loop in ir_loop_list if loop_dim_size[loop] != 1]
                operand_loop_dim[operand]['pr'] = {}
            begin_idx = split_loc + 1

            # Add the dimensionality order of all relevant (including partially relevant) dimensions of this operand
            operand_dimensionality_order[operand] = r_loop_list

        ''' operand_loop_dim_reform remove the pr loop dict, and put the pr-related data dimension (e.g. IX and IY)
         to r and ir dict with "_r" and "_ir" suffix. It brings benefits to loop info extraction after pr loop decoupling step. '''
        operand_loop_dim_reform = deepcopy(operand_loop_dim)
        for operand, dic in operand_loop_dim.items():
            del operand_loop_dim_reform[operand]['pr']
            if dic['pr'] != {}:
                r_extend_list = [pr_data_dim+'_r' for pr_data_dim in pr_loop.keys()]
                ir_extend_list = [pr_data_dim+'_ir' for pr_data_dim in pr_loop.keys()]
                operand_loop_dim_reform[operand]['r'] += r_extend_list
                operand_loop_dim_reform[operand]['ir'] += ir_extend_list

        return operand_loop_dim, operand_loop_dim_reform, operand_list, operand_dimensionality_order

  
    def extract_layer_info(self):
        """
        This function extract basic information for each layer node.
        :return: total_MAC_count, operand_size_elem, operand_size_bit, operand_data_reuse.
        """

        ''' total MAC operation count '''
        total_MAC_count: int = 1
        for ky in self.loop_dim_size:
            total_MAC_count *= self.loop_dim_size[ky]
        self.total_MAC_count = total_MAC_count

        ''' each operand's size (Unit: # of data element) '''
        operand_size_elem: Dict[str, int] = {}
        for operand, relevancy in self.operand_loop_dim.items():
            operand_size_elem[operand] = 1
            for r_loop in relevancy['r']:
                operand_size_elem[operand] *= self.loop_dim_size[r_loop]
            for pr_loop, pr_loop_collect in relevancy['pr'].items():
                multiply_factor = self.calc_tensor_dims(operand, self.loop_dim_size)[pr_loop]
                operand_size_elem[operand] *= multiply_factor
        self.operand_size_elem = operand_size_elem

        ''' each operand's size (Unit: bit) '''
        operand_size_bit: Dict[str, int] = {}
        for operand, size_in_elem in operand_size_elem.items():
            operand_size_bit[operand] = size_in_elem * self.operand_precision[operand]
        self.operand_size_bit = operand_size_bit

        ''' each operand's total data reuse factor, which is total MAC Op/total operand size (in element), 
        i.e. each data element can be used to support how many MAC operation. '''
        operand_data_reuse: Dict[str, float] = {}
        for operand, size_in_elem in operand_size_elem.items():
            operand_data_reuse[operand] = total_MAC_count/size_in_elem
        self.operand_data_reuse = operand_data_reuse

    def get_operand_irrelevant_dimensions(self, layer_op: str):
        """
        Return the irrelevant dimensions of layer operand 'layer_op'.
        """
        return self.operand_loop_dim[layer_op]['ir']

    def get_layer_operand(self, mem_op: str) -> str:
        """
        Return the layer operand associated with the given memory operand for this layer.
        If there is no such memory operand, an error is raised.
        """
        for layer_operand, memory_operand in self.memory_operand_links.items():
            if memory_operand == mem_op:
                return layer_operand
        raise ValueError(f"The memory operand {mem_op} is not present in layer {self}.")

    def get_operand_storage_level(self, layer_op: str):
        """
        Return the memory level at which an input operand is stored.
        If this layer node has no information for the given operand, it returns None.
        """
        if layer_op not in self.source_storage_level:
            return None
        return self.source_storage_level[layer_op]




if __name__ == "__main__":

    equation = 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]'
    dimension_size = {'B': 1, 'K': 32, 'C': 64, 'OY': 28, 'OX': 28, 'FY': 3, 'FX': 3, 'G': 2}
    operand_precision = {'O': 24, 'O_final': 24, 'W': 8, 'I': 8}
    equation_relations = ['ix=ox+fx-1', 'iy=oy+fy-1']

    aa = LayerNode(equation, dimension_size, operand_precision, equation_relations)
    a=1
