import numpy as np
from .._model import DMmodel
from . import param_default

param_default_dict = {
    'signal_freq': 360,
    'Smin': 180,
    'Smax': 300,
    'max_dist': 250,
    'prediction_length': 0,
    'model_momentum': 0,
    'max_vocab': 100,
    'termination_threshold': 0,
    'new_cluster_threshold': 0.3
}


class BeatLex(DMmodel):
    def __init__(self, data_mat, model_name='my_beatlex_model', *args, **para_dict):
        super(BeatLex, self).__init__(data_mat, model_name, *args, **para_dict)
        self.data_mat = data_mat.val_tensor._data

        self.signal_freq = param_default(para_dict, 'signal_freq', param_default_dict)
        self.Smin = param_default(para_dict, 'Smin', param_default_dict)
        self.Smax = param_default(para_dict, 'Smax', param_default_dict)
        self.max_dist = param_default(para_dict, 'max_dist', param_default_dict)
        self.prediction_length = param_default(para_dict, 'prediction_length', param_default_dict)

        self.model_momentum = param_default(para_dict, 'model_momentum', param_default_dict)
        self.max_vocab = param_default(para_dict, 'max_vocab', param_default_dict)
        self.termination_threshold = param_default(para_dict, 'termination_threshold', param_default_dict)
        self.new_cluster_threshold = param_default(para_dict, 'new_cluster_threshold', param_default_dict)

        if 'models' in para_dict.keys():
            self.models = para_dict['models']
        else:
            self.models = []

    def _summarize_sequence(self):
        if len(self.models) == 0:
            start_init_pos = 0
            best_place = self.get_new_segement_size(0)
            end_init_pos = best_place[0] - 1
            print(f'init at {end_init_pos}')
            start_pos_list = [start_init_pos]
            end_pos_list = [end_init_pos]
            self.models.append(self.data_mat[:, start_pos_list[0]:end_pos_list[0]+1])
            idx = [0]
        else:
            start_pos_list = [-1]
            end_pos_list = [-1]
            idx = [-1]
        mean_dev = np.mean((self.data_mat[:] - np.mean(self.data_mat[:])) ** 2)
        best_prefix_length = np.nan
        tot_err = 0
        while end_pos_list[-1] + self.termination_threshold < self.data_mat.shape[1]:
            cur_idx = len(start_pos_list) + 1
            cur_pos = end_pos_list[-1] + 1
            start_pos_list.append(cur_pos)

            num_models = len(self.models)
            ave_costs = np.full((num_models, self.Smax), np.nan)
            cur_end = min(cur_pos + self.Smax - 1, self.data_mat.shape[1])
            if cur_pos == cur_end:
                break
            print('\n========segment {} at {} {}'.format(cur_idx, cur_pos, cur_end))
            data_cur = self.data_mat[:, cur_pos:cur_end]
            for k in range(num_models):
                dtw_dist, dtw_mat, dtw_k, dtw_ways = self.dynamic_time_warping(self.models[k], data_cur)
                dtw_costs = dtw_mat[-1, :]
                ave_costs[k, 0:data_cur.shape[1]] = dtw_costs / np.arange(1, data_cur.shape[1]+1, 1)
                ave_costs[k, 0:self.Smin-1] = np.nan
            best_cost = np.nanmin(ave_costs)
            min_place = np.where(ave_costs == best_cost)
            try:
                min_k = min_place[0][0]
                best_size = min_place[1][0]
            except:
                print('Sequence End Reached')
            if cur_pos + self.Smax >= self.data_mat.shape[1]:
                good_prefix_costs = np.full(num_models, np.nan)
                good_prefix_lengths = np.full(num_models, np.nan)
                for k in range(num_models):
                    _dtw_dist, _dtw_mat, _dtw_k, _dtw_ways = self.dynamic_time_warping(self.models[k], data_cur)
                    prefix_costs = _dtw_mat[:, -1].transpose()
                    ave_prefix_costs = prefix_costs / np.arange(1, len(self.models[k][0])+1, 1)
                    good_prefix_costs[k] = min(ave_prefix_costs)
                    good_prefix_lengths[k] = np.where(ave_prefix_costs == min(ave_prefix_costs))[0]
                best_prefix_cost = min(good_prefix_costs)
                best_prefix_k = np.where(good_prefix_costs == min(good_prefix_costs))[0]
                best_prefix_length = good_prefix_lengths[best_prefix_k]
                print('best prefix found {} {} {}'.format(min_k, best_cost, best_prefix_cost))
                end_pos_list.append(self.data_mat.shape[1])
                if best_prefix_cost > best_cost and len(self.models) < self.max_vocab:
                    print('ending with new cluster')
                    self.models.append(np.array(self.data_mat[:, start_pos_list[-1]:end_pos_list[-1]]))
                    idx.append(num_models)
                    break
                else:
                    print('ending with prefix {}'.format(best_prefix_k))
                    idx.append(best_prefix_k[0])
                    break
            print('cluster cost {}'.format(ave_costs[:, best_size]))
            print('new cluster cost for {}: {}'.format(self.data_mat.shape[0], self.new_cluster_threshold * mean_dev * self.data_mat.shape[0]))
            print('size chosen: {}'.format(best_size+1))
            data_best = self.data_mat[:, cur_pos:cur_pos+best_size+1]
            if best_cost > self.new_cluster_threshold * mean_dev and len(self.models) < self.max_vocab:
                print('new_cluster')
                best_place = self.get_new_segement_size(cur_pos)
                print('best_place: {}'.format(best_place))
                best_S1 = best_place[0]
                print('best_S1: {}'.format(best_S1))
                end_pos_list.append(cur_pos+best_S1)
                idx.append(num_models)
                self.models.append(np.array(self.data_mat[:, start_pos_list[-1]:end_pos_list[-1]+1]))
                print('new cluster starts {} ends {}'.format(start_pos_list[-1], end_pos_list[-1]))
                tot_err += self.new_cluster_threshold * mean_dev * (best_S1 + 1)
            else:
                print('cluster {}'.format(min_k))
                end_pos_list.append(cur_pos+best_size)
                idx.append(min_k)
                tot_err += best_cost*best_size
                _dtw_dist, _dtw_mat, _dtw_k, _dtw_ways = self.dynamic_time_warping(self.models[min_k], data_best)
                trace_summed = np.zeros(self.models[min_k].shape)
                for t in range(0, len(_dtw_ways)):
                    trace_summed[:, _dtw_ways[t][0]] = trace_summed[:, _dtw_ways[t][0]] + data_best[:, _dtw_ways[t][1]]
                trace_keys = list(set([i[0] for i in _dtw_ways]))
                trace_dict = dict.fromkeys(trace_keys, 0)
                for i in _dtw_ways:
                    trace_dict[i[0]] += 1
                trace_counts = list(trace_dict.values())
                trace_aves = trace_summed / trace_counts
                origin_num = idx.count(min_k)
                if self.model_momentum == 0:
                    self.models[min_k] = (origin_num-1) / origin_num * self.models[min_k] + 1/origin_num * trace_aves
                else:
                    self.models[min_k] = self.model_momentum * self.models[min_k] + (1 - self.model_momentum) * trace_aves
        tot_err = tot_err / np.std(self.data_mat, ddof=1) ** 2 + (len(idx) - 1) * np.log2(len(self.data_mat)) + len(idx) * np.log2(len(self.models))
        result = {
            'signal_freq': self.signal_freq,
            'tot_err': tot_err,
            'starts': start_pos_list,
            'ends': end_pos_list,
            'idx': idx,
            'best_prefix_length': best_prefix_length,
            'models': self.models,
        }
        return result

    def get_new_segement_size(self, cur_pos):
        num_models = len(self.models)
        ave_costs = np.full((self.Smax, num_models + 1, self.Smax), np.inf)
        for S in range(self.Smin, self.Smax+1):
            if cur_pos + S >= self.data_mat.shape[1]:
                continue
            for k in range(num_models + 1):
                if k < num_models:
                    cur_model = self.models[k]
                elif k == num_models:
                    cur_model = self.data_mat[:, cur_pos:cur_pos+S]
                data_cur = self.data_mat[:, cur_pos+S:min(self.data_mat.shape[1], cur_pos+S+self.Smax)]
                dist, dtw_mat, ans_k, ans_w = self.dynamic_time_warping(cur_model, data_cur)
                dtw_costs = dtw_mat[-1, :]
                ave_costs[S-1, k, 0:data_cur.shape[1]] = [dtw_costs[index] / (index+1) for index in range(len(dtw_costs))]
                ave_costs[S-1, k, 0:self.Smin] = np.inf
        best_place = np.where(ave_costs == np.nanmin(ave_costs))
        best_place = [x[0] for x in best_place]
        return best_place

    def dynamic_time_warping(self, cur_model, data_seg):
        '''
        cur_model, model chosen from model_list
        data_seg, current handled data
        '''
        rows, N = cur_model.shape
        M = data_seg.shape[1]
        d = self.cal_distance(cur_model, data_seg) / rows
        D = np.full(d.shape, np.inf)
        D[0, 0] = d[0, 0]
        for n in range(1, N):
            D[n, 0] = d[n, 0] + D[n-1, 0]
        for m in range(1, M):
            D[0, m] = d[0, m] + D[0, m-1]
        encode_cost = 1
        mcost = encode_cost * np.std(cur_model, ddof=1) * np.log2(M)
        ncost = encode_cost * np.std(data_seg, ddof=1) * np.log2(N)
        for n in range(1, N):
            m_min = max(1, n-self.max_dist)
            m_max = min(M, n+self.max_dist+1)
            for m in range(m_min, m_max):
                D[n, m] = d[n, m] + min(min(D[n-1, m] + mcost, D[n-1, m-1]), D[n, m-1]+ncost)
        Dist = D[-1, -1]
        n = N-1
        m = M-1
        k = 1
        w = [[n, m]]
        while n + m != 0:
            if n == 0:
                m = m - 1
            elif m == 0:
                n = n - 1
            else:
                num_list = [D[n-1, m], D[n, m-1], D[n-1, m-1]]
                min_val = min(num_list)
                min_pos = num_list.index(min_val)
                if min_pos == 0:
                    n = n - 1
                elif min_pos == 1:
                    m = m - 1
                elif min_pos == 2:
                    n = n - 1
                    m = m - 1
            k = k + 1
            w.append([n, m])
        return Dist, D, k, w

    def cal_distance(self, a, b):
        if a.shape[0] != b.shape[0]:
            raise Exception('A and B should be of same dimensionality')

        ans = np.zeros(shape=(a.shape[1], b.shape[1]))

        for i in range(a.shape[0]):
            _a = np.matrix(a[i])
            _b = np.matrix(b[i])
            aa = np.power(_a, 2)
            bb = np.power(_b, 2)
            ab = np.dot(_a.transpose(), _b)

            plus1 = np.tile(aa.transpose(), (1, bb.shape[1]))
            plus2 = np.tile(bb, (aa.shape[1], 1))
            ans += np.abs(plus1 + plus2 - 2*ab)
        return ans

    def summarization(self, params=None):
        return self._summarize_sequence()

    def run(self):
        return self.summarization()
