"""
the rules of naming records
M: model
R: communication round
B: batch size
E: local epoch
LR: learning rate (step size)
P: the proportion of selected clients in each round
S: random seed
LD: learning rate scheduler + learning rate decay
WD: weight decay
DR: the degree of dropout of clients
AC: the active rate of clients
"""
# import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import os
import ujson
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
# import prettytable as pt


linestyle_tuple = ['dashed', 'dashdot', '-',
                   '--', '-.', ':', 'dotted', 'solid', ]

marker_list = ['o', 's', 'p', '*', 'h', 'X', 'D']

color_list = ['blue', 'green', 'orange', 'red',
              'violet', "teal", "brown", "darkgreen"]

size = 0


def read_data_into_dicts(task, records):
    path = '../fedtask/' + task  # +'/record'
    files = os.listdir(path)
    res = []
    for f in records:
        if f in files:
            file_path = os.path.join(path, f)
            with open(file_path, 'r') as inf:
                rec = ujson.load(inf)
            res.append(rec)
    return res


def draw_curve(dicts, curve='train_losses', legends=[], final_round=-1):
    # plt.figure(figsize=(5,5), dpi=5)
    if not legends:
        legends = [d['meta']['algorithm'] for d in dicts]
    for i, dict in enumerate(dicts):
        num_rounds = dict['meta']['num_rounds']
        eval_interval = dict['meta']['eval_interval']
        x = []
        for round in range(num_rounds + 1):
            if eval_interval > 0 and (round == 0 or round % eval_interval == 0 or round == num_rounds):
                x.append(round)
        if curve == 'train_losses':
            y = [dict[curve][round] for round in range(
                num_rounds + 1) if (round == 0 or round % eval_interval == 0 or round == num_rounds)]
        else:
            y = dict[curve]
        plt.plot(x, y, label=legends[i], linewidth=1,
                 marker=marker_list[i % len(marker_list)],
                 ms=size,
                 linestyle=linestyle_tuple[i % len(linestyle_tuple)],
                 color=color_list[i % (len(color_list))])
        if final_round > 0:
            plt.xlim((0, final_round))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
    plt.legend(loc='best', ncol=1)
    return


def filename_filter(fnames=[], filter={}):
    if filter:
        for key in filter.keys():
            con = filter[key].strip()
            if con[0] in ['[', '{', '(']:
                con = 'in ' + con
            elif '0' <= con[0] <= '9' or con[0] == '.' or con[0] == '-':
                con = '==' + con
            elif 'a' <= con[0] <= 'z' or 'A' <= con[0] <= 'Z':
                con = "'"+con+"'"
            fnames = [f for f in fnames if eval(
                f[f.find('_'+key)+len(key)+1:f.find('_', f.find(key)+1)]+' '+con)]
    return fnames

# def round_to_achieve_test_acc(records, dicts, target=0):
#     tb= pt.PrettyTable()
#     tb.field_names = [
#         'Record',
#         'Round to Achieve {}% Test-Acc.'.format(target),
#     ]
#     for rec, d in zip(records, dicts):
#         r = -1
#         for i in range(len(d['test_accs'])):
#             if d['test_accs'][i]>=target-0.000001:
#                 r = i*d['meta']['eval_interval']
#                 break
#         tb.add_row([rec, r])
#     print(tb)
#     return


def scan_records(task, header='', filter={}):
    path = '../fedtask/' + task  # + '/record'
    files = os.listdir(path)
    # check headers
    files = [f for f in files if f.startswith(header+'_')]
    return filename_filter(files, filter)

# def print_table(records, dicts):
#     tb = pt.PrettyTable()
#     tb.field_names = [
#         'Record',
#         'Test-Acc.',
#         'Valid-Acc.',
#         'Train-Loss',
#         'Test-Loss',
#         'Best Test-Acc./Round',
#         'Highest Valid-Acc.',
#         'Lowest Valid-Acc.',
#         'Mean-Valid-Acc.',
#         'Var-Valid-Acc.',
#     ]
#     for rec,d in zip(records, dicts):
#         testacc  = d['test_accs'][-1]
#         validacc = d['mean_curve'][-1]
#         trainloss = d['train_losses'][-1]
#         testloss = d['test_losses'][-1]
#         bestacc = 0
#         idx = -1
#         for i in range(len(d['test_accs'])):
#             if d['test_accs'][i]>bestacc:
#                 bestacc = d['test_accs'][i]
#                 idx = i*d['meta']['eval_interval']
#         highest = float(np.max(d['valid_accs'][-1]))
#         lowest = float(np.min(d['valid_accs'][-1]))
#         mean_valid = float(np.mean(d['valid_accs'][-1]))
#         var_valid = float(np.std(d['valid_accs'][-1]))
#         tb.add_row([rec, testacc, validacc, trainloss, testloss, str(bestacc)+'/'+str(idx), highest, lowest, mean_valid, var_valid])
#     tb.sortby = 'Test-Acc.'
#     tb.reversesort = True
#     print(tb)


def get_key_from_filename(record, key=''):
    if key == '':
        return ''
    value_start = record.find('_' + key) + len(key) + 1
    value_end = record.find('_', value_start)
    return record[value_start:value_end]


def create_legend(records=[], keys=[]):
    if records == [] or keys == []:
        return records
    res = []
    for rec in records:
        s = [rec[:rec.find('_M')]]
        values = [k + get_key_from_filename(rec, k) for k in keys]
        s.extend(values)
        res.append(" ".join(s))
    return res


def main_func(task, headers, flt):
    # read and filter the filenames
    records = set()
    for h in headers:
        records = records.union(set(scan_records(task, h, flt)))
    records = list(records)
    records.sort()
    # read the selected files into dicts
    dicts = read_data_into_dicts(task, records)

    # print table
    # print_table(records, dicts)

    # draw curves
    curve_names = [
        'train_losses',
        'test_losses',
        'test_accs',
    ]
    # create legends
    legends = create_legend(records, ['P', 'B'])
    for curve in curve_names:
        plt.figure(figsize=(8, 6))
        draw_curve(dicts, curve, legends)
        plt.title(task.split('/')[0].replace('_', ' '), pad=10)
        plt.xlabel("communication rounds")
        plt.ylabel(curve.replace('_', ' '))
        plt.axis('tight')
        ax = plt.gca()
        plt.grid()
        plt.show()
        plt.legend()
        if not Path(f"figures/{task}").exists():
            os.system(f"mkdir -p figures/{task}")
        plt.savefig(f"figures/{task}/{curve}.png")


if __name__ == '__main__':
    # task+record
    headers = [
        # 'scaffold',
        'mp_fedavg',
        # 'mp_fedprox',
        'mp_proposal_4',
        'mp_proposal_4_v3_clustering',
        # 'mp_alg',
        # 'mp_alg_mse',
        # 'feddyn',
        # 'mp_fedkdr',
        # 'mp_fedfsl-mi-adv'
    ]
    flt = {
        # 'E': '8',
        # 'B': '4',
        # 'LR': '0.01',
        # 'R': '200',
        # 'P': '1',
        # 'S': '0',
    }

    num_client = 10
    for s in [1]:
        # task = f'mnist_sparse5_dense5_N{num_client}_K8/mnist/sparse5_dense5/{num_client}client/mnist_sparse5_dense5'
        
        task = f'mnist_cluster_sparse_N{num_client}_K10/mnist/cluster_sparse/{num_client}client/mnist_sparse'
        # task = f'mnist_sparse_N{num_client}_K20/mnist/sparse/{num_client}client/mnist_sparse'
        try:
            main_func(task, headers, flt)
        except ValueError:
            print("error:", task)
