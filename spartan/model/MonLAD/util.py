import numpy as np
import scipy.stats as ss
import pdb
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

def cal_f1(anomalous_node, gt):
    anomalous_node, gt = set(anomalous_node), set(gt)
    right_node = anomalous_node & gt
    P = 0.0
    if len(anomalous_node) != 0:
        P = len(right_node) / len(anomalous_node)
    R = len(right_node) / len(gt)
    F1 = 0.0
    if (P + R) > 0:
        F1 = (2*P*R)/(P+R)
    return F1, P, R

def drawRectbin(xs, ys, outfig=None, xscale='log', yscale='log',
                gridsize=200,
                colorscale=True,
                suptitle='Rectangle binning points',
                xlabel='linear', ylabel='linear'):
    xs = np.array(xs) if type(xs) is list else xs
    ys = np.array(ys) if type(ys) is list else ys

    if xscale == 'log' and min(xs) <= 0:
        print('[Warning] logscale with nonpositive values in x coord')
        print('\tremove {} nonpositives'.format(len(np.argwhere(xs <= 0))))
        xg0 = xs > 0
        xs = xs[xg0]
        ys = ys[xg0]

    if xscale == 'linear' and min(xs) <= 0:
        print('[Warning] logscale with nonpositive values in x coord')
        print('\tremove {} nonpositives'.format(len(np.argwhere(xs <= 0))))
        xg0 = xs > 0
        xs = xs[xg0]
        ys = ys[xg0]

    if yscale == 'log' and min(ys) <= 0:
        print('[Warning] logscale with nonpositive values in y coord')
        print('\tremove {} nonpositives'.format(len(np.argwhere(ys <= 0))))
        yg0 = ys > 0
        xs = xs[yg0]
        ys = ys[yg0]

    fig = plt.figure()

    # color scale
    cnorm = matplotlib.colors.LogNorm() if colorscale else matplotlib.colors.Normalize()
    suptitle = suptitle + ' with a log color scale' if colorscale else suptitle

    # axis space
    if isinstance(gridsize, tuple):
        xgridsize = gridsize[0]
        ygridsize = gridsize[1]
    else:
        xgridsize = ygridsize = gridsize

    if xscale == 'log':
        xlogmax = np.ceil(np.log10(max(xs)))
        x_space = np.logspace(0, xlogmax, xgridsize)
    else:
        x_space = xgridsize
    if yscale == 'log':
        ylogmax = np.ceil(np.log10(max(ys)))
        y_space = np.logspace(0, ylogmax, ygridsize)
    else:
        y_space = ygridsize

    hist = plt.hist2d(xs, ys, bins=(x_space, y_space), cmin=1, norm=cnorm,
            cmap=plt.cm.jet)

    plt.xscale(xscale)
    plt.yscale(yscale)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel(xlabel, font2)
    plt.ylabel(ylabel, font2)
    plt.tight_layout()

    cb = plt.colorbar()
    if colorscale:
        cb.set_label('log10(N)',size=15)
    else:
        cb.set_label('counts',size=15)

    if outfig is not None:
        fig.savefig(outfig,bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    return fig, hist


def boundary(x, k=1.5, existed=False):
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    bound = np.ceil(q75 + k * iqr)
    if existed:
        while bound not in x:
            bound -= 1
    return bound

def get_pareto_score(objscores, alpha=0.9, p=0.9):
    objscores = np.array(objscores) if isinstance(objscores, pd.Series) else objscores
    sortscores = np.array(sorted(objscores))
    sortobjs = np.argsort(objscores)

    tail_fir_score = np.percentile(sortscores, [alpha * 100])[0]

    tailidx = np.argwhere(sortscores >= tail_fir_score)[0][0]
    tailscores = sortscores[tailidx:]
    tailobjs = sortobjs[tailidx:]

    shape, pos, scale = ss.pareto.fit(tailscores)
    cdfs = ss.pareto.cdf(tailscores, shape, pos, scale)

    thre = min(p, max(cdfs))

    levelidxs = np.argwhere(cdfs >= thre)
    levelobjs = tailobjs[levelidxs].T[0]
    if len(objscores[levelobjs]) == 0:
        return -1
    fit_tailscores = min(objscores[levelobjs])
    return fit_tailscores

def call_pareto(df, alpha_c=0.9, p_c=0.9, alpha_i=0.9, p_i=0.9, k=1.5, outpath='', output=False, detect_part=[1,2,3,4], normal=True):

    '''
    :param alpha: fit generalized pareto distribution using (1-alpha)*100% upper tail data
    :param p: given significant value
    :param output:
    :return:
    '''
    assert normal == True
    anomalous_acc = []

    count_list = df['count'].values
    countIn_list = df['IMinusC'].values
    count_list = count_list[count_list>0]
    print('count len:', len(count_list))
    countIn_list = countIn_list[countIn_list>0]
    print('IMinusC len:', len(countIn_list))
    higher_q_C = boundary(count_list, k, existed=True)
    higher_q_I = boundary(countIn_list, k, existed=True)
    print(f'higher_q_C:{higher_q_C}, higher_q_I:{higher_q_I}')

    sum_tail_I, sum_tail_C = 0, 0
    num_tail_I, num_tail_C = 0, 0

    print(f'fit generalized pareto distribution using {(1 - alpha_c) * 100}% (count) and {(1 - alpha_i) * 100}% (IMinusCount) upper tail data')

    count_set = df['count'].value_counts().index
    count_set = np.array(sorted(count_set, reverse=True))
    tmp_m2 = -1
    delta_m2 = 2
    if 1 in detect_part or 3 in detect_part:
        print(f'Part 1: begin find anomalous acc in every count')
        for c in count_set[count_set<=higher_q_C]:
            if c == 0:
                continue
            print(f'count:{c}')
            tmp_I = df[df['count'] == c][['acc_id','IMinusC']]

            tmp_imc = tmp_I['IMinusC']
            fit_tailscores = get_pareto_score(tmp_imc, alpha_c, p_c)

            if len(tmp_imc[tmp_imc>fit_tailscores]) == 0:
                fit_tailscores = tmp_m2
            else:
                if tmp_m2 == -1:
                    tmp_m2 = fit_tailscores
                else:
                    if fit_tailscores > (tmp_m2 + delta_m2):
                        fit_tailscores = tmp_m2
                    else:
                        tmp_m2 = fit_tailscores

            print('fit_tailscores:', fit_tailscores)
            if fit_tailscores == -1:
                print('[Warning] there is not existing a score > p')
                continue

            susp_acc = list(tmp_I[tmp_I['IMinusC'] > max(higher_q_I, fit_tailscores)]['acc_id'])
            if len(susp_acc) > 0:
                print(f'count:{c}, pareto tail score(p={p_c}):{max(higher_q_I, fit_tailscores)}, find count:{len(susp_acc)}')
            anomalous_acc.extend(susp_acc)

            sum_tail_I += fit_tailscores
            num_tail_I += 1

    countIn_set = df['IMinusC'].value_counts().index
    countIn_set = np.array(sorted(countIn_set, reverse=True))
    tmp_c2 = -1
    delta_c2 = 2
    if 2 in detect_part or 4 in detect_part:
        print(f'Part 2: begin find anomalous acc in every IMinusC')
        for i in countIn_set[countIn_set <= higher_q_I]:
            # if i == 0:
            #     continue
            print(f'IMinusC:{i}')
            tmp_C = df[df['IMinusC'] == i][['acc_id', 'count']]
            tmp_c_series = tmp_C['count']

            fit_tailscores = get_pareto_score(tmp_c_series, alpha_i, p_i)
            print('fit_tailscores:', fit_tailscores)

            if len(tmp_c_series[tmp_c_series > fit_tailscores]) == 0:
                fit_tailscores = tmp_c2
            else:
                if tmp_c2 == -1:
                    tmp_c2 = fit_tailscores
                else:
                    if fit_tailscores > (tmp_c2 + delta_c2):
                        fit_tailscores = tmp_c2
                    else:
                        tmp_c2 = fit_tailscores

            print('fit_tailscores:', fit_tailscores)

            if fit_tailscores == -1:
                print('[Warning] there is not existing a score > p')
                continue

            susp_acc = list(tmp_C[tmp_C['count'] > max(higher_q_C, fit_tailscores)]['acc_id'])

            if len(susp_acc) > 0:
                print(f'IMinusC:{i}, pareto tail score(p={p_i}):{max(higher_q_C, fit_tailscores)}, find count:{len(susp_acc)}')
            anomalous_acc.extend(susp_acc)

            if i == 0:
                pdb.set_trace()

            sum_tail_C += fit_tailscores
            num_tail_C += 1

    print(f'Part 3: begin find anomalous acc for remain part')

    # ---- only consider higher_q_C/higher_q_I column

    tail_I_df = df[df['count'] == higher_q_C]['IMinusC']
    print(tail_I_df)
    tail_C_df = df[df['IMinusC'] == higher_q_I]['count']
    print(tail_C_df)

    tail_I = max(get_pareto_score(tail_I_df, alpha_c, p_c), higher_q_I)
    tail_C = max(get_pareto_score(tail_C_df, alpha_i, p_i), higher_q_C)

    assert tail_I != -1
    assert tail_C != -1
    print('tail I:', tail_I)
    print('tail C:', tail_C)

    if 3 in detect_part and tail_I != -1:
        print(f'Part 3.1: begin find anomalous acc in every count with tail I: {tail_I}')
        for c in count_set[count_set > higher_q_C]:
            print(f'count:{c}')
            tmp_I = df[df['count'] == c][['acc_id', 'IMinusC']]
            susp_acc = list(tmp_I[tmp_I['IMinusC'] > tail_I]['acc_id'])
            if len(susp_acc) > 0:
                print(f'count:{c}, find count:{len(susp_acc)}')
            anomalous_acc.extend(susp_acc)

    if 4 in detect_part and tail_C != -1:
        if not normal:
            tail_C = sum_tail_C // num_tail_C
            print(f'sum_tail_C:{sum_tail_C}, num_tail_C:{num_tail_C}, tail_C:{tail_C}')

        print(f'Part 3.2: begin find anomalous acc in every IMinusC with tail C: {tail_C}')
        for i in countIn_set[countIn_set > higher_q_I]:
            print(f'IMinusC:{i}')
            tmp_C = df[df['IMinusC'] == i][['acc_id', 'count']]
            susp_acc = list(tmp_C[tmp_C['count'] > tail_C]['acc_id'])
            if len(susp_acc) > 0:
                print(f'IMinusC:{i}, find count:{len(susp_acc)}')
            anomalous_acc.extend(susp_acc)

    anomalous_acc = set(anomalous_acc)
    print('Total anomalous Acc:', len(anomalous_acc))

    if output:
        detectPartName = "".join(map(str, detect_part))
        anomalous_acc_df = pd.DataFrame()
        anomalous_acc_df['acc_id'] = list(anomalous_acc)
        anomalous_acc_df.to_csv(os.path.join(outpath, 'pareto_acc_'+str(k)+'_'+str(alpha_c)+'_'+str(p_c)+'_'+str(alpha_i)+'_'+str(p_i)+str(10)+f'k_{detectPartName}.csv'), index=False)
        print('save anomalous_acc_df to:', os.path.join(outpath, 'pareto_acc_'+str(k)+'_'+str(alpha_c)+'_'+str(p_c)+'_'+str(alpha_i)+'_'+str(p_i)+str(10)+f'k_{detectPartName}.csv'))

    return anomalous_acc, higher_q_C, higher_q_I, tail_C, tail_I