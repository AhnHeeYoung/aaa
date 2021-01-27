import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score


def read_file(path):
    f = open(path, 'r')
    filename = os.path.splitext(os.path.basename(path))[0]
    file_id = filename.split('_')[0]
    line = f.readlines()
    tsp = float(line[1].split(',')[0])
    label = line[1].split(',')[1].replace('\n', '')
    return filename, file_id, tsp, label


def read_virtualstaining_result(data_dir, pattern='*.txt'):
    result_list = glob.glob(os.path.join(data_dir, pattern))
    result = np.zeros((len(result_list), 4), dtype='object')
    result = pd.DataFrame(result,
                          columns=['filename', 'file_id', 'tsp', 'label'])
    for idx, path in enumerate(result_list):
        result.iloc[idx,] = read_file(path)
    return result


def summary_data(data=None, section='Colon'):
    data['tsp'] = (data['tsp'].values).astype(np.float)
    data = data[['file_id', 'label', 'tsp']].groupby(['file_id', 'label']).mean()
    data = data.reset_index()
    data['predicted_label'] = ['high' if i > 0.65 else 'low' for i in data.tsp.values]
    data = data[['file_id', 'label', 'predicted_label', 'tsp']]
    data.to_csv(os.path.join(root_dir, '{}.csv'.format(section)))


def summary(data):
    data = data[data.label.values != 'int']
    binary_label = [int(i == 'high') for i in data.label.values]
    data.label = np.array(binary_label).astype('float32')
    data.tsp = data.tsp.astype('float32')
    _summary = data[['file_id', 'tsp', 'label']].groupby('file_id').mean()
    _summary = _summary.reset_index()
    return _summary


def tsp_roc_plot(data, section, color):
    data = data[data.label.values != 'int']
    binary_label = [int(i == 'high') for i in data.label.values]
    data.label = np.array(binary_label).astype('float32')
    data.tsp = data.tsp.astype('float32')
    _summary = data[['file_id', 'tsp', 'label']].groupby('file_id').mean()
    _summary = _summary.reset_index()
    tsp = _summary.tsp.values
    label = _summary.label.values
    fpr, tpr, threshold = roc_curve(label, tsp)
    roc_auc = auc(fpr, tpr)
    kappa = cohen_kappa_score((data.tsp > 0.65).astype(int), data.label)

    #  kappa = cohen_kappa_score((data.tsp > np.median(data.tsp)).astype(int), data.label)
    #  plt.plot(fpr, tpr, color,
    #           label='{} AUC = {:0.3} \n {} Kappa(@0.65) = {:0.3}'.format(
    #               section, roc_auc, section,  kappa))
    print('{} : tsp median {}'.format(section, np.median(data.tsp)))
    print('{} : kappa {}'.format(section, kappa))
    plt.plot(fpr, tpr, color,
             label="$AUC_{" + section + "} = " + "{:0.3}$".format(
                    roc_auc))
    plt.legend(loc='lower right', fontsize=8)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel('False Positive Rate', size=10)
    plt.ylabel('True Positive Rate', size=10)


def contingency_table(data, type='None', thresh_hold=0.65):
    data.tsp = data.tsp.astype('float32')
    data = data[['file_id', 'tsp', 'label']].groupby(
        ['file_id', 'label']).mean()
    data.reset_index(inplace=True)
    _real = [i == 'high' for i in data.label.values]
    _pred = data.tsp > thresh_hold
    tmp = pd.DataFrame([_real, _pred], index=['real', 'pred']).transpose()
    print(type)
    print(pd.crosstab(tmp.real, tmp.pred))


root_dir = './data/result/paper'
plt.figure(figsize=(7, 3))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
plt.subplot(1, 2, 2)
#  plt.title('Proposed method with L1(G)^HED', size=12)
plt.title(r'Proposed method with $\mathcal{L}_{L1}(G)^{HED}$', size=10, pad=8)
data_dir = 'dataselect_009unnormal_0903'
result_stomach = read_virtualstaining_result(os.path.join(root_dir, data_dir))
summary_data(result_stomach, 'stomach')
contingency_table(result_stomach, 'proposed stomach')
tsp_roc_plot(data=result_stomach, section='Stomach', color='blue')

data_dir = 'dataselect_colon_009unnormal_0903'
result_colon = read_virtualstaining_result(os.path.join(root_dir, data_dir))
summary_data(result_colon, 'colon')
contingency_table(result_colon, 'proposed colon')
tsp_roc_plot(data=result_colon, section='Colon', color='green')

result_all = pd.concat((result_stomach, result_colon))
contingency_table(result_all, 'proposed stomach+colon')
tsp_roc_plot(data=result_all, section='Stomach+Colon', color='purple')

# plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.title('Proposed method without $\mathcal{L}_{L1}(G)^{HED}$', size=10, pad=8)
data_dir = 'dataselect_standard_0902'
result_stomach = read_virtualstaining_result(os.path.join(root_dir, data_dir))
contingency_table(result_stomach, 'original stmoach')
tsp_roc_plot(data=result_stomach, section='Stomach', color='blue')

data_dir = 'dataselect_colon_standard_0902'
result_colon = read_virtualstaining_result(os.path.join(root_dir, data_dir))
contingency_table(result_colon, 'original colon')
tsp_roc_plot(data=result_colon, section='Colon', color='green')

result_all = pd.concat((result_stomach, result_colon))
contingency_table(result_all, 'proposed stomach+colon')
tsp_roc_plot(data=result_all, section='Stomach+Colon', color='purple')
plt.savefig(os.path.join(root_dir, 'ROC_Curve_standard.eps'), dpi=330, bbox_inches='tight')
plt.savefig(os.path.join(root_dir, 'ROC_Curve_standard.jpg'), dpi=330, bbox_inches='tight')
