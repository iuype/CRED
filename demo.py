

import numpy as np
import mne
from mne.io import read_raw_eeglab
from scipy.io import savemat, loadmat
import h5py

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

    
    for si in range(12):
        ## 读取连续标签
        s = Subjective(
            "200Hz_rawdata/%02d_Subjective.mat" % (si+1),
        )
        # print(s.continous_report.shape)
        # print(s.video_id)
        # print(s.exp_type)
        
        ## 读取脑电数据
        raw = mne.io.read_raw_eeglab(
            "200Hz_rawdata/%02d.set" % (si+1), 
            eog=(), 
            preload=False, 
            uint16_codec=None, 
            montage_units='auto', 
            verbose=None,
        )
        # print(raw)
        # print(mne.events_from_annotations(raw))
        # print(mne.events_from_annotations(raw))
        # print(raw.info["bads"])
        # raw.plot(duration=60, proj=False, n_channels=len(raw.ch_names), remove_dc=False)
        
        
        
        # rawEpoched = mne.Epochs(RawEEGLAB)
        
        
        
        pass