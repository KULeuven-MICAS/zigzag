import unittest
from zigzag.classes.workload.layer_node import LayerNode


class TestLayerNode(unittest.TestCase):

    def test_CONV2D_1(self):

        test_equation = 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]'
        test_loop_dim_size = {'B': 2, 'K': 32, 'C': 64, 'OY': 28, 'OX': 28, 'FY': 3, 'FX': 3, 'G': 4}
        test_operand_precision = {'O': 24, 'O_final': 24, 'W': 8, 'I': 8}
        test_equation_relations = ['ix=ox+fx-1', 'iy=oy+fy-1']
        test_layer_node = LayerNode(test_equation, test_loop_dim_size, test_operand_precision, test_equation_relations)

        result = test_layer_node.operand_loop_dim
        expect = {'O': {'r': ['G', 'B', 'K', 'OY', 'OX'], 'ir': ['C', 'FX', 'FY'], 'pr': {}},
                  'W': {'r': ['G', 'K', 'C', 'FY', 'FX'], 'ir': ['B', 'OX', 'OY'], 'pr': {}},
                  'I': {'r': ['G', 'B', 'C'], 'ir': ['K'], 'pr': {'IY': ['OY', 'FY'], 'IX': ['OX', 'FX']}}}

        for operand in expect:
            self.assertEqual(set(expect[operand]['r']), set(result[operand]['r']))
            self.assertEqual(set(expect[operand]['ir']), set(result[operand]['ir']))
            self.assertEqual(expect[operand]['pr'], result[operand]['pr'])

        pr_loop_size = {'OX': 28, 'FX': 3, 'OY': 28, 'FY': 4}
        result_pr1 = test_layer_node.pr_funcs['IX'](pr_loop_size)
        expect_pr1 = pr_loop_size['OX'] + pr_loop_size['FX'] - 1
        self.assertEqual(expect_pr1, result_pr1)

        result_pr2 = test_layer_node.pr_funcs['IY'](pr_loop_size)
        expect_pr2 = pr_loop_size['OY'] + pr_loop_size['FY'] - 1
        self.assertEqual(expect_pr2, result_pr2)

    def test_CONV2D_2(self):

        test_equation = 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]'
        test_loop_dim_size = {'B': 2, 'K': 32, 'C': 64, 'OY': 28, 'OX': 28, 'FY': 3, 'FX': 3, 'G': 1}
        test_operand_precision = {'O': 24, 'O_final': 24, 'W': 8, 'I': 8}
        test_equation_relations = ['ix=ox+fx-1', 'iy=oy+fy-1']
        test_layer_node = LayerNode(test_equation, test_loop_dim_size, test_operand_precision, test_equation_relations)

        result = test_layer_node.operand_loop_dim
        expect = {'O': {'r': ['B', 'K', 'OY', 'OX'], 'ir': ['C', 'FX', 'FY'], 'pr': {}},
                  'W': {'r': ['K', 'C', 'FY', 'FX'], 'ir': ['B', 'OX', 'OY'], 'pr': {}},
                  'I': {'r': ['B', 'C'], 'ir': ['K'], 'pr': {'IY': ['OY', 'FY'], 'IX': ['OX', 'FX']}}}

        for operand in expect:
            self.assertEqual(set(expect[operand]['r']), set(result[operand]['r']))
            self.assertEqual(set(expect[operand]['ir']), set(result[operand]['ir']))
            self.assertEqual(expect[operand]['pr'], result[operand]['pr'])

        pr_loop_size = {'OX': 14, 'FX': 3, 'OY': 15, 'FY': 4}
        result_pr1 = test_layer_node.pr_funcs['IX'](pr_loop_size)
        expect_pr1 = pr_loop_size['OX'] + pr_loop_size['FX'] - 1
        self.assertEqual(expect_pr1, result_pr1)

        result_pr2 = test_layer_node.pr_funcs['IY'](pr_loop_size)
        expect_pr2 = pr_loop_size['OY'] + pr_loop_size['FY'] - 1
        self.assertEqual(expect_pr2, result_pr2)

    def test_CONV2D_stride2(self):

        test_equation = 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]'
        test_loop_dim_size = {'B': 2, 'K': 32, 'C': 64, 'OY': 28, 'OX': 28, 'FY': 3, 'FX': 3, 'G': 1}
        test_operand_precision = {'O': 24, 'O_final': 24, 'W': 8, 'I': 8}
        test_equation_relations = ['ix=2*(ox-1)+2*(fx-1)+1', 'iy=2*(oy-1)+2*(fy-1)+1']
        test_layer_node = LayerNode(test_equation, test_loop_dim_size, test_operand_precision, test_equation_relations)

        result = test_layer_node.operand_loop_dim
        expect = {'O': {'r': ['B', 'K', 'OY', 'OX'], 'ir': ['C', 'FX', 'FY'], 'pr': {}},
                  'W': {'r': ['K', 'C', 'FY', 'FX'], 'ir': ['B', 'OX', 'OY'], 'pr': {}},
                  'I': {'r': ['B', 'C'], 'ir': ['K'], 'pr': {'IY': ['OY', 'FY'], 'IX': ['OX', 'FX']}}}

        for operand in expect:
            self.assertEqual(set(expect[operand]['r']), set(result[operand]['r']))
            self.assertEqual(set(expect[operand]['ir']), set(result[operand]['ir']))
            self.assertEqual(expect[operand]['pr'], result[operand]['pr'])

        pr_loop_size = {'OX': 14, 'FX': 3, 'OY': 15, 'FY': 4}
        result_pr1 = test_layer_node.pr_funcs['IX'](pr_loop_size)
        expect_pr1 = 2 * (pr_loop_size['OX'] - 1) + 2 * (pr_loop_size['FX'] - 1) + 1
        self.assertEqual(expect_pr1, result_pr1)

        result_pr2 = test_layer_node.pr_funcs['IY'](pr_loop_size)
        expect_pr2 = 2 * (pr_loop_size['OY'] - 1) + 2 * (pr_loop_size['FY'] - 1) + 1
        self.assertEqual(expect_pr2, result_pr2)

    def test_CONV1D(self):

        test_equation = 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]'
        test_loop_dim_size = {'B': 2, 'K': 32, 'C': 64, 'OY': 28, 'OX': 1, 'FY': 3, 'FX': 1, 'G': 1}
        test_operand_precision = {'O': 24, 'O_final': 24, 'W': 8, 'I': 8}
        test_equation_relations = ['ix=ox+fx-1', 'iy=oy+fy-1']
        test_layer_node = LayerNode(test_equation, test_loop_dim_size, test_operand_precision, test_equation_relations)

        result = test_layer_node.operand_loop_dim
        expect = {'O': {'r': ['B', 'K', 'OY'], 'ir': ['C', 'FY'], 'pr': {}},
                  'W': {'r': ['K', 'C', 'FY'], 'ir': ['B', 'OY'], 'pr': {}},
                  'I': {'r': ['B', 'C'], 'ir': ['K'], 'pr': {'IY': ['OY', 'FY']}}}

        for operand in expect:
            self.assertEqual(set(expect[operand]['r']), set(result[operand]['r']))
            self.assertEqual(set(expect[operand]['ir']), set(result[operand]['ir']), )
            self.assertEqual(expect[operand]['pr'], result[operand]['pr'])

        pr_loop_size = {'OY': 28, 'FY': 3}
        result_pr2 = test_layer_node.pr_funcs['IY'](pr_loop_size)
        expect_pr2 = pr_loop_size['OY'] + pr_loop_size['FY'] - 1
        self.assertEqual(expect_pr2, result_pr2)

    def test_DW(self):

        test_equation = 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]'
        test_loop_dim_size = {'B': 1, 'K': 1, 'C': 1, 'OY': 28, 'OX': 28, 'FY': 3, 'FX': 3, 'G': 256}
        test_operand_precision = {'O': 24, 'O_final': 24, 'W': 8, 'I': 8}
        test_equation_relations = ['ix=ox+fx-1', 'iy=oy+fy-1']
        test_layer_node = LayerNode(test_equation, test_loop_dim_size, test_operand_precision, test_equation_relations)

        result = test_layer_node.operand_loop_dim
        expect = {'O': {'r': ['G', 'OY', 'OX'], 'ir': ['FX', 'FY'], 'pr': {}},
                  'W': {'r': ['G', 'FY', 'FX'], 'ir': ['OX', 'OY'], 'pr': {}},
                  'I': {'r': ['G'], 'ir': [], 'pr': {'IY': ['OY', 'FY'], 'IX': ['OX', 'FX']}}}

        for operand in expect:
            self.assertEqual(set(expect[operand]['r']), set(result[operand]['r']))
            self.assertEqual(set(expect[operand]['ir']), set(result[operand]['ir']))
            self.assertEqual(expect[operand]['pr'], result[operand]['pr'])

        pr_loop_size = {'OX': 14, 'FX': 3, 'OY': 15, 'FY': 4}
        result_pr1 = test_layer_node.pr_funcs['IX'](pr_loop_size)
        expect_pr1 = pr_loop_size['OX'] + pr_loop_size['FX'] - 1
        self.assertEqual(expect_pr1, result_pr1)

        result_pr2 = test_layer_node.pr_funcs['IY'](pr_loop_size)
        expect_pr2 = pr_loop_size['OY'] + pr_loop_size['FY'] - 1
        self.assertEqual(expect_pr2, result_pr2)

    def test_PW(self):

        test_equation = 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]'
        test_loop_dim_size = {'B': 1, 'K': 32, 'C': 64, 'OY': 28, 'OX': 28, 'FY': 1, 'FX': 1, 'G': 1}
        test_operand_precision = {'O': 24, 'O_final': 24, 'W': 8, 'I': 8}
        test_equation_relations = ['ix=ox+fx-1', 'iy=oy+fy-1']
        test_layer_node = LayerNode(test_equation, test_loop_dim_size, test_operand_precision, test_equation_relations)

        result = test_layer_node.operand_loop_dim
        expect = {'O': {'r': ['K', 'OY', 'OX'], 'ir': ['C'], 'pr': {}},
                  'W': {'r': ['K', 'C'], 'ir': ['OX', 'OY'], 'pr': {}},
                  'I': {'r': ['C', 'OX', 'OY'], 'ir': ['K'], 'pr': {}}}

        for operand in expect:
            self.assertEqual(set(expect[operand]['r']), set(result[operand]['r']))
            self.assertEqual(set(expect[operand]['ir']), set(result[operand]['ir']))
            self.assertEqual(expect[operand]['pr'], result[operand]['pr'])

    def test_FC(self):

        test_equation = 'O[g][b][k][oy][ox]+=W[g][k][c][fy][fx]*I[g][b][c][ix][iy]'
        test_loop_dim_size = {'B': 1, 'K': 32, 'C': 64, 'OY': 1, 'OX': 1, 'FY': 1, 'FX': 1, 'G': 1}
        test_operand_precision = {'O': 24, 'O_final': 24, 'W': 8, 'I': 8}
        test_equation_relations = ['ix=ox+fx-1', 'iy=oy+fy-1']
        test_layer_node = LayerNode(test_equation, test_loop_dim_size, test_operand_precision, test_equation_relations)

        result = test_layer_node.operand_loop_dim
        expect = {'O': {'r': ['K'], 'ir': ['C'], 'pr': {}},
                  'W': {'r': ['K', 'C'], 'ir': [], 'pr': {}},
                  'I': {'r': ['C'], 'ir': ['K'], 'pr': {}}}

        for operand in expect:
            self.assertEqual(set(expect[operand]['r']), set(result[operand]['r']))
            self.assertEqual(set(expect[operand]['ir']), set(result[operand]['ir']))
            self.assertEqual(expect[operand]['pr'], result[operand]['pr'])

    def test_MVM(self):

        test_equation = 'Y[i]+=A[i][j]*B[j]'
        test_loop_dim_size = {'I': 56, 'J': 112}
        test_operand_precision = {'Y': 24, 'Y_final': 24, 'A': 8, 'B': 8}
        test_layer_node = LayerNode(test_equation, test_loop_dim_size, test_operand_precision)

        result = test_layer_node.operand_loop_dim
        expect = {'Y': {'r': ['I'], 'ir': ['J'], 'pr': {}},
                  'A': {'r': ['I', 'J'], 'ir': [], 'pr': {}},
                  'B': {'r': ['J'], 'ir': ['I'], 'pr': {}}}

        for operand in expect:
            self.assertEqual(set(expect[operand]['r']), set(result[operand]['r']))
            self.assertEqual(set(expect[operand]['ir']), set(result[operand]['ir']))
            self.assertEqual(expect[operand]['pr'], result[operand]['pr'])

    def test_GEMM(self):

        test_equation = 'Y[i][k]+=A[i][j]*B[j][k]'
        test_loop_dim_size = {'I': 56, 'J': 112, 'K': 28}
        test_operand_precision = {'Y': 24, 'Y_final': 24, 'A': 8, 'B': 8}
        test_layer_node = LayerNode(test_equation, test_loop_dim_size, test_operand_precision)

        result = test_layer_node.operand_loop_dim
        expect = {'Y': {'r': ['I', 'K'], 'ir': ['J'], 'pr': {}},
                  'A': {'r': ['I', 'J'], 'ir': ['K'], 'pr': {}},
                  'B': {'r': ['J', 'K'], 'ir': ['I'], 'pr': {}}}

        for operand in expect:
            self.assertEqual(set(expect[operand]['r']), set(result[operand]['r']))
            self.assertEqual(set(expect[operand]['ir']), set(result[operand]['ir']))
            self.assertEqual(expect[operand]['pr'], result[operand]['pr'])

    def test_MMc(self):

        test_equation = 'Y[i][j]+=A[i][k]*B[k][l]*C[l][j]'
        test_loop_dim_size = {'I': 56, 'J': 112, 'K': 28, 'L': 14}
        test_operand_precision = {'Y': 24, 'Y_final': 24, 'A': 8, 'B': 8, 'C': 8}
        test_layer_node = LayerNode(test_equation, test_loop_dim_size, test_operand_precision)

        result = test_layer_node.operand_loop_dim
        expect = {'Y': {'r': ['I', 'J'], 'ir': ['K', 'L'], 'pr': {}},
                  'A': {'r': ['I', 'K'], 'ir': ['J', 'L'], 'pr': {}},
                  'B': {'r': ['K', 'L'], 'ir': ['I', 'J'], 'pr': {}},
                  'C': {'r': ['L', 'J'], 'ir': ['I', 'K'], 'pr': {}}}

        for operand in expect:
            self.assertEqual(set(expect[operand]['r']), set(result[operand]['r']))
            self.assertEqual(set(expect[operand]['ir']), set(result[operand]['ir']))
            self.assertEqual(expect[operand]['pr'], result[operand]['pr'])

    def test_MTTKRP(self):

        test_equation = 'Y[i][j]+=A[i][k][l]*B[k][j]*C[l][j]'
        test_loop_dim_size = {'I': 56, 'J': 112, 'K': 28, 'L': 14}
        test_operand_precision = {'Y': 24, 'Y_final': 24, 'A': 8, 'B': 8, 'C': 8}
        test_layer_node = LayerNode(test_equation, test_loop_dim_size, test_operand_precision)

        result = test_layer_node.operand_loop_dim
        expect = {'Y': {'r': ['I', 'J'], 'ir': ['K', 'L'], 'pr': {}},
                  'A': {'r': ['I', 'K', 'L'], 'ir': ['J'], 'pr': {}},
                  'B': {'r': ['K', 'J'], 'ir': ['I', 'L'], 'pr': {}},
                  'C': {'r': ['L', 'J'], 'ir': ['I', 'K'], 'pr': {}}}

        for operand in expect:
            self.assertEqual(set(expect[operand]['r']), set(result[operand]['r']))
            self.assertEqual(set(expect[operand]['ir']), set(result[operand]['ir']))
            self.assertEqual(expect[operand]['pr'], result[operand]['pr'])