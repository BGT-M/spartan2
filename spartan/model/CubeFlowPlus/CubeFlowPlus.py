from .._model import DMmodel
from .res_util import *
from ...util.basicutil import param_default
from .. import CubeFlow


class CubeFlowPlus(DMmodel):
    def __init__(self, tensorList:list):
        self.amt_tensor, self.cmt_tensor = tensorList[0], tensorList[1]
    
    def __str__(self):
        return str(vars(self))
    
    def run(self, **params):
        self.res = param_default(params, 'cf_res', None)
        self.handle_biggraph_type = param_default(params, 'handle_biggraph_type', 1)
        self.max_node_limit = param_default(params, 'max_node_limit', 100)
        self.maxsize1 = param_default(params, 'maxsize1', -1)
        self.maxsize2 = param_default(params, 'maxsize2', (-1, 100, -1))
        self.outpath1 = param_default(params, 'outpath1', '') # CubeFlow's result without limit
        self.outpath2 = param_default(params, 'outpath2', '') 
        self.alpha = param_default(params, 'alpha', 0.8)
        self.k = param_default(params, 'k', 1)
        self.dim = param_default(params, 'dim', 3)
        self.del_type = param_default(params, 'del_type', 1)
        self.is_find_all_blocks = param_default(params, 'is_find_all_blocks', False)
        
        self.initData()
        
        if self.res is None:    
            self.res = self.call_cubeflow()

        # 步骤一：划分大图为连通子图
        subgraph_connected_list, subgraph_timebin_list, subgraph_timebin_idx_list, subgraph_can_divided = \
            divide_connected_conponents(self.res, self.amt_tensor, self.cmt_tensor)

        # 步骤二：对每个大图（总节点数超过最大限制）进行单独处理
        # 选择1：去掉度最大的节点，放入连通图算法
        # 选择2：运行带约束的CubeFlow，直至跑空所有节点为止

        subgraph_connected_list, subgraph_timebin_list, subgraph_timebin_idx_list = \
            handle_big_graph(subgraph_connected_list, subgraph_timebin_list, subgraph_timebin_idx_list,
                             subgraph_can_divided, self.max_node_limit,
                             self.amt_tensor, self.cmt_tensor, self.handle_biggraph_type, self.maxsize2, self.alpha, self.dim, self.del_type)

        print(len(subgraph_connected_list))

        # 步骤三：对连通子图列表打分并排序

        subgraph_score_list, subgraph_score1_list, subgraph_score2_list, amct_list = \
            score_connected_graph_list(subgraph_connected_list, subgraph_timebin_list, subgraph_timebin_idx_list,
                                       self.alpha, self.amt_stensor, self.cmt_stensor, self.amt_tensor, self.cmt_tensor)

        subgraph_score_list = np.array(subgraph_score_list)
        subgraph_score_sort_index_list = np.argsort(subgraph_score_list)[::-1]  # 分数由高到低排序
        res_new = []

        for idx in subgraph_score_sort_index_list:
            res_new.append([amct_list[idx], subgraph_score_list[idx]])
            
        if self.outpath2 != '':
            for i in range(len(res_new)):
                saveres2txt(res_new[i], self.outpath2)
    
            print(f'save new result to path:{self.outpath2} success!')
        
        return res_new, subgraph_score_list, subgraph_score1_list, subgraph_score2_list, amct_list           
        
    def initData(self):
        self.amt_stensor = self.amt_tensor.toSTensor(hasvalue=True)
        self.cmt_stensor = self.cmt_tensor.toSTensor(hasvalue=True)
        print(self.amt_stensor.shape)
        print(self.cmt_stensor.shape)
        
    
    def call_cubeflow(self):
        cf = CubeFlow([self.amt_stensor, self.cmt_stensor], alpha=self.alpha, k=self.k, dim=self.dim, outpath=self.outpath1)
        res = cf.run(del_type=self.del_type, maxsize=self.maxsize1, is_find_all_blocks=self.is_find_all_blocks)
        return res
    
    
        
        
    
