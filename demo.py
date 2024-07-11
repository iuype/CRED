

import numpy as np
import mne
from mne.io import read_raw_eeglab
from scipy.io import savemat, loadmat
import h5py
import neurokit2 as nk
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import warnings
from scipy.io import savemat
import matplotlib.pyplot as plt
# 忽略所有的UserWarning
warnings.filterwarnings('ignore', category=UserWarning)

class Subjective:

    def __init__(self, filename=None):

        m = h5py.File(filename)
        n_trial = m['Subjective']['continous_report'].shape[0]

        self.continous_report = np.zeros((n_trial, 1800))
        self.video_id = np.zeros((n_trial, ))
        self.exp_type = [0] * n_trial
        for i in range(n_trial):
            self.continous_report[i, :] = np.array(m[m['Subjective']['continous_report'][i,0]])
            self.video_id[i] = np.array(m[m['Subjective']['video_id'][i,0]])
            self.exp_type[i] = m[m['Subjective']['type'][i,0]].shape[0]
        self.exp_type = ['WATCH' if x > 4 else 'RATE' for x in self.exp_type]

if __name__ == "__main__":

    print(nk.cite())
    acc = []
    for si in range(0, 24):

        ## 读取连续标签
        s = Subjective(
            "1000Hz_rawdata_multimodal/%02d_Subjective.mat" % (si+1),
        )

        ## 读取脑电数据
        raw = mne.io.read_raw_eeglab(
            "1000Hz_rawdata_multimodal/%02d.set" % (si+1), 
            eog=(), 
            preload=False, 
            uint16_codec=None, 
            montage_units='auto', 
            verbose=None,
        )
        rawEpoched = mne.Epochs(
            raw,
            tmin=-1,
            tmax=60,
            baseline=(None, 0.0),
        ).load_data()
        print(raw.info, rawEpoched._data.shape)
        
        X = rawEpoched._data[:,-6:, :] # ['PPG', 'RSP', 'EMG-A', 'EDA', 'ECG', 'EMG-B']
        y = rawEpoched.events[:, -1]

        for i in range(X.shape[0]):

            ecg_cleaned = nk.ecg_clean(X[i, 4, :])
            peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=1000)
            if i == 0:
                HRV = nk.hrv(peaks, sampling_rate=1000, show=False)
            else:
                HRV = pd.concat([HRV, nk.hrv(peaks, sampling_rate=1000, show=False)], ignore_index=True)
                

        cols_with_nan = HRV.isna().any()
        print(cols_with_nan[cols_with_nan].index.tolist())
        HRV = HRV.drop(cols_with_nan[cols_with_nan].index, axis=1)
        X_HRV = np.array(HRV)
        w=np.isinf(X_HRV)
        X_HRV = X_HRV[:, ~w.any(axis=0)]
        X_HRV_scaled = StandardScaler().fit_transform(X_HRV)
        
        print(X.shape, y.shape)
        
        X = X_HRV_scaled
        
        # 指定留1样本的交叉验证
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.8, random_state=0)
        
        # 用于比较模型的分类器，这里以RandomForestClassifier为例
        model = svm.SVC(kernel='rbf', C=1.0)
        
        score_list = []
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score_list.append((model.score(X_test, y_test)))

        with open('acc.log', 'a') as f:
            f.write(
                "ACC=%.2f\n" % (np.array(score_list).mean()) 
            )