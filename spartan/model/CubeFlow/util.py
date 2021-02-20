import numpy as np
from collections import defaultdict
import time
import sparse as sp
import os
from ...backend import STensor
from ...tensor import Graph


def mkdir_self(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def preprocess_data(stensorList:list, dim):
    print('Dim: ', dim)
    mt_dict, mt_m_dict, concise_a_mt_dict, concise_c_mt_dict = {}, {}, defaultdict(int), defaultdict(int)
    mt_t_dict = {}
    mt_d4_dict = {}  # k_symbols
    
    amt_stensor = stensorList[0]
    cmt_stensor = stensorList[1]
    amt_coo, cmt_coo = amt_stensor._data, cmt_stensor._data
    a_mt_coo = amt_coo.reshape(shape=(amt_coo.shape[0], -1))  # combine M and T
    c_mt_coo = cmt_coo.reshape(shape=(cmt_coo.shape[0], -1))  # combine M and T
    print(a_mt_coo)
    print(c_mt_coo)

    a_mt_nnz = a_mt_coo.nonzero()
    c_mt_nnz = c_mt_coo.nonzero()

    stt = time.time()
    if dim == 3:
        m_shape_a = amt_stensor.shape[2]
        m_shape_c = cmt_stensor.shape[2]
        for a, mt, elem in zip(a_mt_nnz[0], a_mt_nnz[1], a_mt_coo.data):
            if mt not in mt_dict:
                mt_dict[mt] = len(mt_dict)
            map_m = mt_dict[mt]
            if map_m not in mt_m_dict:
                mt_m_dict[map_m] = mt // m_shape_a
            if map_m not in mt_t_dict:
                mt_t_dict[map_m] = mt % m_shape_a
            key = (a, map_m)
            concise_a_mt_dict[key] += elem
        print('time:', time.time() - stt)
        for c, mt, elem in zip(c_mt_nnz[0], c_mt_nnz[1], c_mt_coo.data):
            if mt not in mt_dict:
                mt_dict[mt] = len(mt_dict)
            map_m = mt_dict[mt]
            if map_m not in mt_m_dict:
                mt_m_dict[map_m] = mt // m_shape_c
            if map_m not in mt_t_dict:
                mt_t_dict[map_m] = mt % m_shape_c
            key = (c, map_m)
            concise_c_mt_dict[key] += elem
        print('time:', time.time() - stt)
    elif dim == 4:
        t_shape_a = amt_stensor.shape[2]
        t_shape_c = cmt_stensor.shape[2]
        d4_shape_a = amt_stensor.shape[3]
        d4_shape_c = cmt_stensor.shape[3]
        for a, mt, elem in zip(a_mt_nnz[0], a_mt_nnz[1], a_mt_coo.data):
            if mt not in mt_dict:
                mt_dict[mt] = len(mt_dict)
            map_m = mt_dict[mt]
            if map_m not in mt_m_dict:
                mt_m_dict[map_m] = mt // (t_shape_a * d4_shape_a)
            tmp = mt % (t_shape_a * d4_shape_a)
            if map_m not in mt_t_dict:
                mt_t_dict[map_m] = tmp // d4_shape_a
            if map_m not in mt_d4_dict:
                mt_d4_dict[map_m] = tmp % d4_shape_a
            key = (a, map_m)
            concise_a_mt_dict[key] += elem
        print('time:', time.time() - stt)
        for c, mt, elem in zip(c_mt_nnz[0], c_mt_nnz[1], c_mt_coo.data):
            if mt not in mt_dict:
                mt_dict[mt] = len(mt_dict)
            map_m = mt_dict[mt]
            if map_m not in mt_m_dict:
                mt_m_dict[map_m] = mt // (t_shape_c * d4_shape_c)
            tmp = mt % (t_shape_a * d4_shape_a)
            if map_m not in mt_t_dict:
                mt_t_dict[map_m] = tmp // d4_shape_c
            if map_m not in mt_d4_dict:
                mt_d4_dict[map_m] = tmp % d4_shape_c
            key = (c, map_m)
            concise_c_mt_dict[key] += elem
        print('time:', time.time() - stt)
    else:
        raise Exception('dim should be 3 or 4!')

    concise_a_mt_coo = sp.COO(concise_a_mt_dict, shape=(a_mt_coo.shape[0], len(mt_dict)))
    concise_c_mt_coo = sp.COO(concise_c_mt_dict, shape=(c_mt_coo.shape[0], len(mt_dict)))
    print(concise_a_mt_coo)
    print(concise_c_mt_coo)

    concise_a_mt_stensor = STensor((concise_a_mt_coo.coords, concise_a_mt_coo.data))
    concise_c_mt_stensor = STensor((concise_c_mt_coo.coords, concise_c_mt_coo.data))

    a_mt_graph = Graph(graph_tensor=concise_a_mt_stensor, bipartite=True, weighted=True)
    c_mt_graph = Graph(graph_tensor=concise_c_mt_stensor, bipartite=True, weighted=True)
    
    print('preprocess Data completed!')
    return [a_mt_graph, c_mt_graph], [mt_m_dict, mt_t_dict, mt_d4_dict]

def find_true_m(res, dim, mt_dictList):     
    res_M = []
    res_t = []
    res_d4 = []
    for m in res[0][1]:
        res_M.append(mt_dictList[0][m])
        res_t.append(mt_dictList[1][m])
        if dim == 4:
            res_d4.append(mt_dictList[2][m])
    res_M = list(set(res_M))
    res_t = list(set(res_t))
    if dim == 4:
        res_d4 = list(set(res_d4))
        return res_M, res_t, res_d4
    return res_M, res_t

def get_real_res(res, dim, mt_dictList, outpath):
    if dim == 3:
        res_M, res_t = find_true_m(res, dim, mt_dictList)
    elif dim == 4:
        res_M, res_t, res_d4 = find_true_m(res, dim, mt_dictList)
    else:
        raise Exception('Dim ERR! should be 3 or 4.')
    import copy
    real_res = copy.deepcopy(res)
    real_res[0][1] = res_M
    real_res[0].append(res_t)
    if dim == 4:
        real_res[0].append(res_d4)
    print('Find block:')
    if dim == 3:
        print(len(real_res[0][0]), len(real_res[0][1]), len(real_res[0][2]), len(real_res[0][3]))
    else:
        print(len(real_res[0][0]), len(real_res[0][1]), len(real_res[0][2]), len(real_res[0][3]), len(real_res[0][4]))
    saveres2txt(real_res, outpath)
    return real_res

def saveres2npy(res, outpath):
    mkdir_self(outpath)
    for i in range(len(res[0])):
        res[0][i] = list(res[0][i])
    nres = []
    resAddr = outpath+'res.npy'
    if os.path.exists(resAddr):
        nres = np.load(resAddr, allow_pickle=True)
        nres = nres.tolist()
    nres.extend([res])
    np.save(resAddr, nres)
    print('res save completed!')

def saveres2txt(res, outpath):
    if outpath == '':
        return 
    mkdir_self(outpath)
    for i in range(len(res[0])):
        res[0][i] = list(res[0][i])
    with open(outpath+"res.txt","a") as f:
        f.write('[')
        f.write('\n')
        for i in range(len(res[0])):
            for j in range(len(res[0][i])):
                f.write(str(res[0][i][j]))
                f.write('\t')
            f.write('\n')
        f.write(str(res[1]))
        f.write('\n')
        f.write(']')
        f.write('\n')
    print('save res.txt success!')

def loadtxt2res(outpath, dim):
    res = []
    tmp_res = []
    with open(outpath+"res.txt","r") as f:
        contents =f.readlines()
        for line in contents:
            if line.startswith('['):
                tmp_res = []
                continue
            if line.startswith(']'):
                res.append(tmp_res)
                continue
            acc_list = line.strip().split("\t")
            if len(tmp_res) < dim:
                acc_list = list(map(int, acc_list))
                tmp_res.append(acc_list)
            else:
                score = float(acc_list[0])
                tmp_res.append(score)
    print('load res.txt success!')
    return res
        
        
    
    
    
    
