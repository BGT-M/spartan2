import re
import pandas as pd
import numpy as np
import networkx as nx
import os
import pdb
import sparse
import scipy.sparse as ssp
import timeit

from ...tensor import TensorData
from .. import CubeFlow

# Remember to add spartan to you PATH
# import sys
# sys.path.append("/<dir to spartan2>/spartan2")
# sys.path.append("./spartan2-FCC")
# import spartan as st

def divide_connected_conponents(res, amt_tensor, cmt_tensor):
    subgraph_connected_list = []
    subgraph_timebin_list = []  # 用于查询子张量的timebin
    subgraph_timebin_idx_list = []  # 存放每个子张量时间维度在subgraph_timebin_list的下标
    subgraph_can_divided = []
    # find all suspicious edges
    for top_i in range(len(res)):
        idx1 = list(res[top_i][0][0])  # A
        idx2 = list(res[top_i][0][1])  # M
        idx3 = list(res[top_i][0][2])  # C
        time_bins = list(res[top_i][0][3])  # timebins
        subgraph_timebin_list.append(time_bins)

        susp_amt_df = amt_tensor.data.loc[amt_tensor.data[0].isin(idx1) & amt_tensor.data[1].isin(idx2) & amt_tensor.data[2].isin(time_bins)]
        susp_cmt_df = cmt_tensor.data.loc[cmt_tensor.data[0].isin(idx3) & cmt_tensor.data[1].isin(idx2) & cmt_tensor.data[2].isin(time_bins)]
        susp_amt_df.columns = ['a', 'm', 't', 'money']
        susp_cmt_df.columns = ['c', 'm', 't', 'money']
        susp_amt_df['a'] = susp_amt_df['a'].apply(lambda x: 'a' + str(x))
        susp_amt_df['m'] = susp_amt_df['m'].apply(lambda x: 'm' + str(x))
        susp_cmt_df['c'] = susp_cmt_df['c'].apply(lambda x: 'c' + str(x))
        susp_cmt_df['m'] = susp_cmt_df['m'].apply(lambda x: 'm' + str(x))

        # 构造有向图（多条边）
        G = nx.MultiDiGraph()
        G.add_weighted_edges_from(susp_amt_df[['a', 'm', 'money']].values)
        G.add_weighted_edges_from(susp_cmt_df[['m', 'c', 'money']].values)

        # 获得连通子图
        subgraph_connected_list_topi = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
        subgraph_connected_list.extend(subgraph_connected_list_topi)
        subgraph_timebin_idx_list.extend([top_i] * len(subgraph_connected_list_topi))

        # 维护判断是否可拆解list
        if res[top_i][1] == float("-inf") and len(subgraph_connected_list_topi) == 1:
            subgraph_can_divided.append(0) # 不可拆分

        else:
            subgraph_can_divided.extend([1] * len(subgraph_connected_list_topi))


    return subgraph_connected_list, subgraph_timebin_list, subgraph_timebin_idx_list, subgraph_can_divided

def divide_amc_dict(node_list):
    '''
    用于将str账号混合列表处理回int形式，并划分为A、M、C
    '''
    a_list = []
    m_list = []
    c_list = []
    for acc in node_list:
        acc_int = int(re.sub('[a-zA-Z]', "", acc)) # 转换为int型
        if acc.startswith('a'):
            a_list.append(acc_int)
        elif acc.startswith('m'):
            m_list.append(acc_int)
        elif acc.startswith('c'):
            c_list.append(acc_int)
        else:
            raise ValueError
    return a_list, m_list, c_list

def get_csv_from_subgraph(node_list, amt_tensor, cmt_tensor, subgraph_timebin_list, timebin_idx):
    '''
    从字符串账号列表中获得适用于算法输入的csv数据
    '''
    a_list, m_list, c_list = divide_amc_dict(node_list)
    time_bins =  subgraph_timebin_list[timebin_idx]
    am_subgraph_df = amt_tensor.data[(amt_tensor.data[0].isin(a_list)) & (amt_tensor.data[1].isin(m_list)) & (amt_tensor.data[2].isin(time_bins))]
    cm_subgraph_df = cmt_tensor.data[(cmt_tensor.data[0].isin(c_list)) & (cmt_tensor.data[1].isin(m_list)) & (cmt_tensor.data[2].isin(time_bins))]
    return am_subgraph_df, cm_subgraph_df

def relabel_acct(am_df, cm_df):
    '''
    对账号和timebin进行重新编号，加快算法运行速度
    '''
    temp_left = am_df.copy()
    temp_right = cm_df.copy()
    temp_left.columns = ['a_acct', 'm_acct', 'time_bin', 'amount']
    temp_right.columns = ['c_acct', 'm_acct', 'time_bin', 'amount']

    middle_acc_list = pd.Series(list(set(temp_left['m_acct']) & set(temp_right['m_acct']))) \
        .drop_duplicates().sort_values().reset_index(drop=True)
    temp_left = temp_left[temp_left['m_acct'].isin(middle_acc_list.values)]
    temp_right = temp_right[temp_right['m_acct'].isin(middle_acc_list.values)]

    timebin_list = temp_left['time_bin'].append(temp_right['time_bin']).drop_duplicates().sort_values().reset_index(
        drop=True)

    print('middle acct len: ', len(middle_acc_list))
    print('timebin count:', len(timebin_list))

    left_acc_list = temp_left['a_acct'].drop_duplicates().sort_values().reset_index(drop=True)  # 构建字典
    right_acc_list = temp_right['c_acct'].drop_duplicates().sort_values().reset_index(drop=True)

    # map accounts from 0
    left_acc_dict = dict(zip(list(left_acc_list), list(left_acc_list.index.astype(int))))
    middle_acc_dict = dict(zip(list(middle_acc_list), list(middle_acc_list.index.astype(int))))
    right_acc_dict = dict(zip(list(right_acc_list), list(right_acc_list.index.astype(int))))  # 构建字典完成
    timebin_dict = dict(zip(list(timebin_list), list(timebin_list.index.astype(int))))

    temp_left['a_acct'] = temp_left['a_acct'].apply(lambda x: left_acc_dict[x])  # 对原有属性进行映射
    temp_left['m_acct'] = temp_left['m_acct'].apply(lambda x: middle_acc_dict[x])
    temp_left['time_bin'] = temp_left['time_bin'].apply(lambda x: timebin_dict[x])
    temp_right['m_acct'] = temp_right['m_acct'].apply(lambda x: middle_acc_dict[x])
    temp_right['c_acct'] = temp_right['c_acct'].apply(lambda x: right_acc_dict[x])
    temp_right['time_bin'] = temp_right['time_bin'].apply(lambda x: timebin_dict[x])

    # map 0 to original account
    left_acc_dict2 = dict(zip(list(left_acc_list.index.astype(int)), list(left_acc_list)))
    middle_acc_dict2 = dict(zip(list(middle_acc_list.index.astype(int)), list(middle_acc_list)))
    right_acc_dict2 = dict(zip(list(right_acc_list.index.astype(int)), list(right_acc_list)))  # 构建字典完成
    timebin_dict2 = dict(zip(list(timebin_list.index.astype(int)), list(timebin_list)))

    temp_left.columns = temp_right.columns = [0, 1, 2, 3]

    return temp_left, temp_right, [left_acc_dict2, middle_acc_dict2, right_acc_dict2, timebin_dict2]

def call_cubeflow(am_df, cm_df, maxsize, outpath='', alpha=0.8, k=-1, dim=3, del_type=1, is_find_all_blocks=True):
    temp_left = am_df.copy()
    temp_right = cm_df.copy()

    temp_left, temp_right, dict_list = relabel_acct(temp_left, temp_right)

    amt_tensor = TensorData(temp_left)
    cmt_tensor = TensorData(temp_right)

    amt_stensor = amt_tensor.toSTensor(hasvalue=True)
    cmt_stensor = cmt_tensor.toSTensor(hasvalue=True)
    print(amt_stensor.shape)
    print(cmt_stensor.shape)

    cf = CubeFlow([amt_stensor, cmt_stensor], alpha=alpha, k=k, dim=dim, outpath=outpath)
    res = cf.run(del_type=del_type, maxsize=maxsize, is_find_all_blocks=is_find_all_blocks)

    for i in range(len(res)):
        a_list = list(map(lambda x: dict_list[0][x], res[i][0][0]))
        m_list = list(map(lambda x: dict_list[1][x], res[i][0][1]))
        c_list = list(map(lambda x: dict_list[2][x], res[i][0][2]))
        t_list = list(map(lambda x: dict_list[3][x], res[i][0][3]))
        res[i][0][0] = a_list
        res[i][0][1] = m_list
        res[i][0][2] = c_list
        res[i][0][3] = t_list

    return res

def cal_score_from_graph(subgraph, alpha):
    # 连通子图的am+cm矩阵
    subG_weight_list = list(subgraph.edges(data='weight', default=1))
    subG_weight_df = pd.DataFrame(subG_weight_list)
    print(subG_weight_df)

    subG_in_Df = subG_weight_df[subG_weight_df[1].str.contains('m')]
    subG_out_Df = subG_weight_df[subG_weight_df[0].str.contains('m')]
    subG_indegree = subG_in_Df.groupby(1).sum()
    subG_outdegree = subG_out_Df.groupby(0).sum()

    subG_indegree.index.name = 'm'
    subG_outdegree.index.name = 'm'
    subG_indegree.columns = ['indegree']
    subG_outdegree.columns = ['outdegree']

    subG_info = pd.concat([subG_indegree, subG_outdegree], axis=1, sort=False)
    subG_info['indegree'].fillna(0, inplace=True)
    subG_info['outdegree'].fillna(0, inplace=True)

    #     # 连通子图中m层节点
    #     m_layer_nodes = pd.concat([subG_in_Df[1], subG_out_Df[0]]).drop_duplicates(keep='first', inplace=False)
    #     print('m_layer_nodes len:', len(m_layer_nodes))
    #     m_node_num.append(len(m_layer_nodes))

    # 计算得分
    subG_info_array = subG_info.to_numpy()
    subG_info['f'], subG_info['q'] = np.min(subG_info_array, axis=1), np.max(subG_info_array, axis=1)
    subG_info['r'] = subG_info['q'] - subG_info['f']
    print(subG_info)

    curScore1 = subG_info['f'].sum()
    curScore2 = subG_info['r'].sum()
    s = subgraph.number_of_nodes()
    curScore = ((1 - alpha) * curScore1 - alpha * curScore2) / s
    print(f'score of g(S): {curScore}, f: {curScore1}, q-f: {curScore2}')

    return curScore

def revert_acct(amct_list, dict_dir):
    '''
    用于将a、m、c账户映射回原始字符串编号
    :param a_list: 需要映射的a账户list
    :param m_list: ...
    :param c_list: ...
    :param dict_dir: 存放映射关系文件的目录
    :return:
    '''
    a_list = amct_list[0]
    m_list = amct_list[1]
    c_list = amct_list[2]
    t_list = amct_list[3]

    dic_a = pd.read_csv(os.path.join(dict_dir, 'dict_a.csv'))
    dic_m = pd.read_csv(os.path.join(dict_dir, 'dict_m.csv'))
    dic_c = pd.read_csv(os.path.join(dict_dir, 'dict_c.csv'))
    dic_t = pd.read_csv(os.path.join(dict_dir, 'dict_time.csv'))

    dict_a = dict(zip(dic_a['new'], dic_a['ori']))  # mapped_number: origin_number
    dict_m = dict(zip(dic_m['new'], dic_m['ori']))
    dict_c = dict(zip(dic_c['new'], dic_c['ori']))
    dict_t = dict(zip(dic_t['new'], dic_t['old']))


    ori_a_list = list(map(lambda x: dict_a[x], a_list))
    ori_m_list = list(map(lambda x: dict_m[x], m_list))
    ori_c_list = list(map(lambda x: dict_c[x], c_list))
    ori_t_list = list(map(lambda x: dict_t[x], t_list))

    return [ori_a_list, ori_m_list, ori_c_list, ori_t_list]

def cal_score_from_acc(amct_list, alpha, amt_tensor, cmt_tensor):
    a_list, m_list, c_list, t_list = amct_list[0], amct_list[1], amct_list[2], amct_list[3]
    # 连通子图的am+cm矩阵
    subG_in_Df = amt_tensor.data[amt_tensor.data[0].isin(a_list) & amt_tensor.data[1].isin(m_list) & amt_tensor.data[2].isin(t_list)]
    subG_out_Df = cmt_tensor.data[cmt_tensor.data[0].isin(c_list) & cmt_tensor.data[1].isin(m_list) & cmt_tensor.data[2].isin(t_list)]
    subG_indegree = subG_in_Df[[1, 2, 3]].groupby([1, 2]).sum()
    subG_outdegree = subG_out_Df[[1, 2, 3]].groupby([1, 2]).sum()

    subG_indegree.columns = ['indegree']
    subG_outdegree.columns = ['outdegree']

    subG_info = pd.concat([subG_indegree, subG_outdegree], axis=1, sort=False)
    subG_info['indegree'].fillna(0, inplace=True)
    subG_info['outdegree'].fillna(0, inplace=True)

    print(subG_info)

    # 计算得分
    subG_info_array = subG_info.to_numpy()
    subG_info['f'], subG_info['q'] = np.min(subG_info_array, axis=1), np.max(subG_info_array, axis=1)
    subG_info['r'] = subG_info['q'] - subG_info['f']
    print(subG_info)

    curScore1 = subG_info['f'].sum()
    curScore2 = subG_info['r'].sum()

    time_df = subG_in_Df[2].append(subG_out_Df[2]).drop_duplicates().sort_values().reset_index(drop=True)
    t_len = time_df.shape[0]

    s = t_len + len(m_list) + len(a_list) + len(c_list)
    curScore = ((1 - alpha) * curScore1 - alpha * curScore2) / s
    print(f'score of g(S): {curScore}, f: {curScore1}, q-f: {curScore2}, s:{s}')

    return curScore, curScore1, curScore2

def cal_score_from_acc_sparse(amct_list, alpha, csr_matrix_list, timebin2idx_dict):
    a_list, m_list, c_list, t_list = amct_list[0], amct_list[1], amct_list[2], amct_list[3]
    t_len, curScore1, curScore2 = 0, 0, 0
    for t in t_list:
        am_matrix = csr_matrix_list[timebin2idx_dict[t]][0]
        am_matrix_sub = am_matrix[a_list, :].tocsc()[:, m_list]
        cm_matrix = csr_matrix_list[timebin2idx_dict[t]][1]
        cm_matrix_sub = cm_matrix[c_list, :].tocsc()[:, m_list]

        if am_matrix_sub.nnz == 0 and cm_matrix_sub.nnz == 0:
            continue
        else:
            t_len += 1

        indegree_arr = np.squeeze(am_matrix_sub.sum(axis=0).A)
        outdegree_arr = np.squeeze(cm_matrix_sub.sum(axis=0).A)

        if indegree_arr.shape == ():
            indegree_arr = np.array([indegree_arr.tolist()])
            outdegree_arr = np.array([outdegree_arr.tolist()])

        mid_min = []
        for m1, m2 in zip(indegree_arr, outdegree_arr):
            mid_min.append(min(m1, m2))  # fi

        curScore1 += sum(mid_min)
        curScore2 += sum(abs(indegree_arr - outdegree_arr))

    s = t_len + len(m_list) + len(a_list) + len(c_list)
    curScore = ((1 - alpha) * curScore1 - alpha * curScore2) / s
    print(f'score of g(S): {curScore}, f: {curScore1}, q-f: {curScore2}, s:{s}')
    return curScore, curScore1, curScore2

def cal_score_from_acc_sparse_fs(amct_list, alpha, a_mt_mat, c_mt_mat):
    a_list, m_list, c_list, t_list = amct_list[0], amct_list[1], amct_list[2], amct_list[3]

    a_mt_mat_sub = a_mt_mat[a_list, :].tocsc()[:, m_list]
    c_mt_mat_sub = c_mt_mat[c_list, :].tocsc()[:, m_list]

    indegree_arr = np.squeeze(a_mt_mat_sub.sum(axis=0).A)
    outdegree_arr = np.squeeze(c_mt_mat_sub.sum(axis=0).A)

    if indegree_arr.shape == ():
        indegree_arr = np.array([indegree_arr.tolist()])
        outdegree_arr = np.array([outdegree_arr.tolist()])

    mid_min = []
    for m1, m2 in zip(indegree_arr, outdegree_arr):
        mid_min.append(min(m1, m2))  # fi

    curScore1, curScore2 = 0, 0

    curScore1 += sum(mid_min)
    curScore2 += sum(abs(indegree_arr - outdegree_arr))

    s = len(m_list) + len(a_list) + len(c_list)
    curScore = ((1 - alpha) * curScore1 - alpha * curScore2) / s
    # print(f'score of g(S): {curScore}, f: {curScore1}, q-f: {curScore2}, s:{s}')
    return curScore, curScore1, curScore2

def get_zero_matrix(idx, size):
    row = np.array(list(range(size)))
    col = np.array(list(range(size)))
    data = np.array([1]*size)
    row = np.delete(row, idx)
    col = np.delete(col, idx)
    data = np.delete(data, idx)
    return ssp.csr_matrix((data, (row, col)), shape=(size, size))

def loadtxt2res(outpath, dim, data_type=int):
    res = []
    tmp_res = []
    with open(outpath, "r") as f:
        contents =f.readlines()
        for line in contents:
            if line.startswith('['):
                tmp_res = []
                continue
            if line.startswith(']'):
                res.append(tmp_res)
                continue
            acc_list = line.strip().split("\t")
            if len(tmp_res) < dim + 1:
                acc_list = list(map(data_type, acc_list))
                tmp_res.append(acc_list)
            else:
                score = float(acc_list[0])
                tmp_res = [tmp_res, score]
    print('load res.txt success!')
    return res

def handle_big_graph(subgraph_connected_list, subgraph_timebin_list, subgraph_timebin_idx_list, subgraph_can_divided, max_node_limit,
                     amt_tensor, cmt_tensor, handle_biggraph_type, maxsize2, alpha, dim, del_type):
    # 步骤二：对每个大图（总节点数超过最大限制）进行单独处理
    # 选择1：去掉度最大的节点，放入连通图算法
    # 选择2：运行带约束的CubeFlow，直至跑空所有节点为止
    curr_i = 0
    while curr_i < len(subgraph_connected_list):
        curr_subgraph = subgraph_connected_list[curr_i]
        curr_nodes_list = list(curr_subgraph.nodes())
        a_list_tmp, m_list_tmp, c_list_tmp = divide_amc_dict(curr_nodes_list)
        if len(m_list_tmp) > max_node_limit:

            if handle_biggraph_type == 1:
                m_str_list_tmp = ['m' + str(m) for m in m_list_tmp]
                curr_degree_dict = dict(curr_subgraph.degree(m_str_list_tmp))  # 去掉m账户中度最大的节点

                max_degree_node = max(curr_degree_dict, key=curr_degree_dict.get)
                curr_matrix = nx.to_scipy_sparse_matrix(curr_subgraph, curr_nodes_list)

                node_idx = curr_nodes_list.index(max_degree_node)

                zero_matrix = get_zero_matrix(node_idx, curr_matrix.shape[0])
                curr_matrix = zero_matrix * curr_matrix * zero_matrix  # 删除结点
                curr_subgraph = nx.from_scipy_sparse_matrix(curr_matrix,
                                                            create_using=nx.MultiDiGraph)

                mapping = {}
                for i in range(len(curr_nodes_list)):
                    mapping[i] = curr_nodes_list[i]

                curr_subgraph = nx.relabel_nodes(curr_subgraph, mapping)
                curr_subgraph.remove_node(max_degree_node)
                print(f'max degree node [{max_degree_node}] has been removed!')

                # 重新运行连通子图算法
                curr_subgraph_connected_list = [curr_subgraph.subgraph(c).copy() for c in
                                                nx.weakly_connected_components(curr_subgraph)]
                print('add subgraph num:', len(curr_subgraph_connected_list))

                subgraph_connected_list.extend(curr_subgraph_connected_list)
                tb_idx = subgraph_timebin_idx_list[curr_i]
                subgraph_timebin_idx_list.append(tb_idx)

            elif handle_biggraph_type == 2:

                if subgraph_can_divided[curr_i] == 0:
                    curr_i += 1
                    continue

                # 准备算法输入数据
                am_df, cm_df = get_csv_from_subgraph(curr_nodes_list, amt_tensor, cmt_tensor, subgraph_timebin_list,
                                                     subgraph_timebin_idx_list[curr_i])

                # 调用CubeFlow算法
                curr_res = call_cubeflow(am_df, cm_df, maxsize=maxsize2, outpath='', alpha=alpha, k=-1, dim=dim,
                                         del_type=del_type, is_find_all_blocks=True)
                print('find subgraph num:', len(curr_res))

                # 复用拆分连通子图代码
                curr_subgraph_connected_list, curr_subgraph_timebin_list, curr_subgraph_timebin_idx_list, curr_subgraph_can_divided \
                    = divide_connected_conponents(curr_res, amt_tensor, cmt_tensor)
                print('add subgraph num:', len(curr_subgraph_connected_list))

                # 加入现有的连通子图list中
                subgraph_connected_list.extend(curr_subgraph_connected_list)
                old_timebin_list_len = len(subgraph_timebin_list)
                curr_subgraph_timebin_idx_list = list(
                    map(lambda x: x + old_timebin_list_len, curr_subgraph_timebin_idx_list))

                subgraph_timebin_idx_list += curr_subgraph_timebin_idx_list
                subgraph_timebin_list += curr_subgraph_timebin_list

                subgraph_can_divided += curr_subgraph_can_divided

                del subgraph_can_divided[curr_i]

            else:
                raise ValueError

            del subgraph_timebin_idx_list[curr_i]
            del subgraph_connected_list[curr_i]


        else:
            curr_i += 1

    return subgraph_connected_list, subgraph_timebin_list, subgraph_timebin_idx_list

def sparse_df_to_csr(df, matrix_shape):
    tensor = TensorData(df)
    stensor = tensor.toSTensor(hasvalue=True)
    tmp = stensor._data
    tmp = ssp.coo_matrix((tmp.data, (tmp.coords[0], tmp.coords[1])), shape=matrix_shape)
    return tmp.tocsr()

def prepare_for_sparse_score(subgraph_timebin_list, amt_stensor, cmt_stensor, amt_tensor, cmt_tensor):
    subgraph_timebin_list_flatten = []
    for i in range(len(subgraph_timebin_list)):
        subgraph_timebin_list_flatten.extend(subgraph_timebin_list[i])
    subgraph_timebin_list_flatten = list(set(subgraph_timebin_list_flatten))

    am_matrix_shape = amt_stensor.shape[:2]
    cm_matrix_shape = cmt_stensor.shape[:2]
    print(am_matrix_shape, cm_matrix_shape)

    csr_matrix_list = []
    for i in range(len(subgraph_timebin_list_flatten)):
        am_df = amt_tensor.data[amt_tensor.data[2] == subgraph_timebin_list_flatten[i]][[0, 1, 3]]
        cm_df = cmt_tensor.data[cmt_tensor.data[2] == subgraph_timebin_list_flatten[i]][[0, 1, 3]]

        am_matrix = sparse_df_to_csr(am_df, am_matrix_shape)
        cm_matrix = sparse_df_to_csr(cm_df, cm_matrix_shape)
        csr_matrix_list.append([am_matrix, cm_matrix])

    timebin2idx_dict = {}
    for i in range(len(subgraph_timebin_list_flatten)):
        timebin2idx_dict[subgraph_timebin_list_flatten[i]] = i

    return csr_matrix_list, timebin2idx_dict

def score_connected_graph_list(subgraph_connected_list, subgraph_timebin_list, subgraph_timebin_idx_list,
                               alpha, amt_stensor, cmt_stensor, amt_tensor, cmt_tensor):
    print('begin to score connected subgraph....')

    # csr_matrix_list, timebin2idx_dict = prepare_for_sparse_score(subgraph_timebin_list, amt_stensor, cmt_stensor,
    #                                                              amt_tensor, cmt_tensor)

    # 算法函数使用fs目标函数，即不包含t时使用
    a_mt_mat = amt_stensor.sum(axis=2)._data.tocsr()
    c_mt_mat = cmt_stensor.sum(axis=2)._data.tocsr()

    subgraph_score_list = []
    subgraph_score1_list = []  # f
    subgraph_score2_list = []  # q - f
    amct_list = []
    total_len = len(subgraph_connected_list)

    for idx in range(len(subgraph_connected_list)):
        # print(f'subgraph {idx}/{total_len}: ', end='')
        subgraph = subgraph_connected_list[idx]
        a_list_tmp, m_list_tmp, c_list_tmp = divide_amc_dict(list(subgraph.nodes()))

        amct_list_sub = [a_list_tmp, m_list_tmp, c_list_tmp, subgraph_timebin_list[subgraph_timebin_idx_list[idx]]]
        # curScore, curScore1, curScore2 = cal_score_from_acc_sparse(amct_list_sub, alpha, csr_matrix_list,
        #                                                            timebin2idx_dict)

        curScore, curScore1, curScore2 = cal_score_from_acc_sparse_fs(amct_list_sub, alpha, a_mt_mat, c_mt_mat)

        subgraph_score_list.append(curScore)
        subgraph_score1_list.append(curScore1)
        subgraph_score2_list.append(curScore2)
        amct_list.append(amct_list_sub)

    return subgraph_score_list, subgraph_score1_list, subgraph_score2_list, amct_list

def mkdir_self(dir):
    if dir.endswith('.txt'):
        dir_list = dir.split('/')
        dir = '/'.join(dir_list[:-1])

    if not os.path.exists(dir):
        os.makedirs(dir)

def saveres2txt(res, outpath):
    mkdir_self(outpath)
    for i in range(len(res[0])):
        res[0][i] = list(res[0][i])
    with open(outpath, "a") as f:
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
    # print('save res.txt success!')

def loadlisttxt2res(outpath):
    res = []
    with open(outpath, "r") as f:
        contents =f.readlines()
        for line in contents:
            res.append(str(line.strip()))
    print('load res.txt success!')
    return res
