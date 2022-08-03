from .._model import DMmodel
from .ZeroOutCore import ZeroOutCore
from .ZeroOutCoreCFD import ZeroOutCoreCFD
import pdb
import pandas as pd
import numpy as np
import os
from .util import call_pareto

class MonLAD(DMmodel):
    def __init__(self, stream_tensor, **param_dict):
        self.stream_tensor = stream_tensor
        self.window = param_dict['window']
        self.stride = param_dict['stride']
        self.ts_colidx = param_dict['ts_idx']
        self.has_edge = param_dict['has_edge']
        self.deltaUp = param_dict['deltaUp']
        self.deltaDown = param_dict['deltaDown']
        self.epsilon = param_dict['epsilon']
        if param_dict.get('source_type'):
            source_type = param_dict['source_type']
        else:
            source_type = 'VYDAJ'
        if param_dict.get('des_type'):
            des_type = param_dict['des_type']
        else:
            des_type = 'PRIJEM'

        self.count_df = None

        if self.has_edge:
            self.core = ZeroOutCore(self.deltaUp, self.deltaDown, self.epsilon)
        else:
            self.core = ZeroOutCoreCFD(self.deltaUp, self.deltaDown, self.epsilon, source_type, des_type)

    def anomaly_detection(self, detect_part=[1, 2, 3, 4], alpha=0.98, k=1.5, p=0.99, outpath=None):
        if self.count_df is None:
            self.run()
        count_df = self.count_df
        count_df['IMinusC'] = count_df['countIn'] - count_df['count']
        count_df = count_df[count_df['count'] > 0]
        anomalous_acc, higher_q_B, higher_q_F, tail_B, tail_F = call_pareto(count_df, alpha_c=alpha, p_c=p, alpha_i=alpha, p_i=p, k=k, outpath=outpath,
                                output=False,
                                detect_part=detect_part)  # 1: part1; 2: part3; 3: part2-1; 4:part2-2

        if outpath:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            np.save(os.path.join(outpath, 'res.npy'), anomalous_acc)

        return anomalous_acc

    def run(self):
        while True:
            try:
                tensorlist = self.stream_tensor.fetch_slide_window(self.window, self.stride, self.ts_colidx, decode=False)
                # print(tensorlist)
                # pdb.set_trace()
                tensor_arr = tensorlist.values
                for i in range(len(tensor_arr)):
                    if self.has_edge:
                        acc_id_source, acc_id_des, timestamp, weight = tensor_arr[i][0], tensor_arr[i][1], tensor_arr[i][2], int(tensor_arr[i][3])
                        acc_count, acc_countIn = self.core(acc_id_source, acc_id_des, timestamp, weight)
                    else:
                        acc_id, timestamp, transaction_type, weight = tensor_arr[i][0], tensor_arr[i][1], tensor_arr[i][2], int(tensor_arr[i][3])
                        acc_count, acc_countIn = self.core(acc_id, transaction_type, weight) # acc_id, transaction_type, weight
                        if acc_count == -1:
                            continue

            except Exception as e:
                print(e)
                break
        
        count_df = pd.DataFrame()
        count_df['acc_id'] =  np.array(list(self.core.countDict.keys()))
        count_df['count'] = np.array(list(self.core.countDict.values()))
        count_df['countIn'] = np.array(list(self.core.countInDict.values()))
        self.count_df = count_df

        return count_df
