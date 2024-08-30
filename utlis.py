"""
contains definitions for the classes wrapping the production of the torch loaders
"""
from scipy.integrate import simpson as simps
import numpy as np
import mne
import torch
from torch.utils.data import TensorDataset , DataLoader ,random_split
# Set the log level to WARNING to suppress informational messages
# Set the log level to WARNING to suppress informational messages
mne.set_log_level('WARNING')
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg


class Utlis:
    @staticmethod
    def gen_bins(psd,freq,bin_dict,use_simp=False):
        """
        helper function of get_psds
        
        Args:
            psd: psds returned by psd_array_welch
            freq : freq returned by psd_array_welch
            simps: if true area under each bin is calculated using simps and relative power for each bin is returned
                else the power of freqs in each bin are summed and divided by total number of freqs 

            bin_dict: dictionary that defined the number and range of individual bins. Bins are constructed by taking average of values in given range.
                    required struture of items
                        key(string) -> name of the bin (alpha,beta,gamma,etc)
                        value (list) -> [start_freq,end_freq] , both last and start freq are included
        Returns:
            bin_arr: numpy ndarray of shape (num_bin,num_channels)

                        
        """

        bin_list = [] # list to store all bin values

        # matching the range provided with neariest available frequency
        for key,items in bin_dict.items():
            
            f_start, f_end = items[0],items[1]
            
            assert f_start > 0, f"start range for bin names{key} can not be lesser that zero"
            assert f_end > 0 or f_end == -1, f"end range for bin names{key} should be greater than zero or -1 (to include all points after start)"
            
            # obtaining nearest freq vals to given range
            idx_start = np.argmin(abs(freq - f_start))

            if f_end != -1:
                # print("hit if")
                idx_end = np.argmin(abs(freq - f_end))
            else:
                # print("hit else")
                idx_end = freq.shape[0]
            

            # aggregating the values

            if use_simp == True:
                # finding approximate area for the current band
                dx = freq[1]-freq[0] # interval between sampling
                bin_value = simps(y=psd[:,idx_start:idx_end],dx=dx)
            else:    

                sum_acc_chns = np.sum(psd[:,idx_start:idx_end],axis=1) # taking sum of freqs accross channels
                bin_value = sum_acc_chns /(idx_end - idx_start)  # dividing by the lenght of the interval
            
            bin_list.append(bin_value) # appending the bin values in last dim of value of given key
        
        if simps == True:
            # finding the total area under the curve
            total_power = simps(psd,dx=dx)
            bin_list = bin_dict/total_power # returning the relative power per signal

        return np.array(bin_list)

        
    @staticmethod
    def get_ar_coefs(raw_eeg_signal,AR_order = 4, duration=1):
        """ 
        returns the Auto Regressive coefs of the from given raw eeg signal, duration given between 0 and 1 is the percentage of signal length used for AR extractions

        Args:
            raw_eeg_signal: ndarray in form of n_samples * n_channels * n _points
            duration: the percentage of the signal to be used for feature extraction (default -> 1 i.e full signal)
        """

        assert AR_order > 1 , "ar order should be greate than 1"
        assert duration > 0 and duration <= 1 , f"duration should be between 0 and 1, represents percentage"

        # subset ot eeg signal used for obtaning ar_coefs
        sub_points = int(raw_eeg_signal.shape[-1]*duration)
        sub_eeg_signal = raw_eeg_signal[:,:,0:sub_points]


        master_ar_list =[] # list containing ar coefs for every sample

        # print("eeg.shape ",eeg.shape)

        for sample in sub_eeg_signal:

            # print("sample.shape ",sample.shape)

            sample_ar_list = []

            # looping over channels

            for idx in range(sample.shape[0]):
                chn_eeg = sample[idx] 
                # print("chn_eeg.shape ",chn_eeg.shape)
                # break;

                ar_model = AutoReg(chn_eeg, lags=AR_order)
                ar_fit = ar_model.fit() 
                
                # Get the AR coefficients
                ar_coefficients = ar_fit.params
                sample_ar_list.append(ar_coefficients)

            np_sample_ar = np.array(sample_ar_list)
            vec_sample_ar = np_sample_ar.reshape(-1,)
            master_ar_list.append(vec_sample_ar)
        
        return np.array(master_ar_list)


    @staticmethod
    def get_psd(raw_eeg_signal,duration=1,bin_dict=None,use_simp=True):
        """
        returns the PSD of given data, duration given between 0 and 1 is the percentage of signal length used for psd extraction 
        
        Args:
            raw_eeg_signal: ndarray in form of n_samples * n_channels * n _points
            duration: the percentage of the signal to be used for feature extraction (default -> 1 i.e full signal)
            bin_dict: dictionary that defined the number and range of individual bins. Bins are constructed by taking average of values in given range.
                    required struture of items
                        key(string) -> name of the bin (alpha,beta,gamma,etc)
                        value (list) -> [start_freq,end_freq] , both last and start freq are included
        """
        
        assert duration > 0 and duration <= 1 , f"duration should be between 0 and 1, represents percentage"

        sub_points = int(raw_eeg_signal.shape[-1]*duration)



        best_fft = 80 # no of points to be used in calculation of fast fourier transform
        curr_fft= best_fft

        if sub_points < curr_fft:
            curr_fft = sub_points
            print("Signal length is too small, using lesser points for fft")


        # sub_eeg = raw_eeg_signal[:,:,0:sub_points]


        features_list = [] # list containing all of the psd values for each sample
        for sample in sub_eeg:
        
            psds, freqs = mne.time_frequency.psd_array_welch(sample, fmin=0, fmax=np.inf,sfreq=160, n_fft=curr_fft)

            if bin_dict is not None:
                # binning the frequencies by taking average
                psds = Utlis.gen_bins(psds,freqs,bin_dict,use_simp=use_simp) # shape (num_bin,num_channels)
                

            # print("psds.shape ",psds.shape)
            vec_psds = psds.reshape(-1,)  # vectorizing the psd features into one column 
            features_list.append(vec_psds) 

        arr = np.array(features_list)  

        return  [arr , freqs]

    @staticmethod
    def split_arrays(x, y,split_size=0.2,keep_distribution=False):

        """
        randomly splits given array while maintaing the cross pairing between arrays passed.
        to ensure that data is evenly distributed among classes, it makes sures that each split follows same
        distribution of data, despite of its size

        Args:

            x : arr1
            y:  arr2 if(keep_distribution is True then obtain labels from y)
            split_size : between 0 and 1 represents the split ratio 
            keep_distribtion: if true then maintains the same data distribution across classes

        Returns: 

            x_sub_1 : first portion of x
            x_sub_2 : second portion of x
            y_sub_1 : first portion of y
            y_sub_2 : second portion of y

        """

        assert len(x) == len(y), "x and y must have the same length"
        assert split_size > 0 and split_size < 1 ,"split_size must be between 0 and 1"

        if keep_distribution == True: 

            unique_classes, class_indices = np.unique(y,return_inverse=True) # finding indices of unique class labels

            # inititalizing four sub_arrs to be returned

            y_shape = list(y.shape)
            y_shape[0] = 0

            x_shape = list(x.shape)
            x_shape[0] = 0
            



            y_sub_1 = np.empty(y_shape)
            y_sub_2 = np.empty(y_shape)
            x_sub_1 = np.empty(x_shape)
            x_sub_2 = np.empty(x_shape)

            for cls in unique_classes:
                # looping over all class_indices
                elem_indicies = np.where(y==cls)[0] # finding the indicies of the element belonging to that class

                # recursively calling split_arrays() again to split class specific elements


                x_subset_1, x_subset_2, y_subset_1, y_subset_2  = Utlis.split_arrays(x[elem_indicies],y[elem_indicies],split_size=split_size,keep_distribution=False)

                # extending the previous results with new one
                
                x_sub_1 = np.concatenate((x_subset_1,x_sub_1),axis=0) 
                x_sub_2 = np.concatenate((x_subset_2,x_sub_2),axis=0) 
                y_sub_1 = np.concatenate((y_subset_1,y_sub_1),axis=0) 
                y_sub_2 = np.concatenate((y_subset_2,y_sub_2),axis=0) 

            
            # shuffiling indicies so that classes are also mixed

            indices_x_sub_1 = np.arange(len(x_sub_1))
            np.random.shuffle(indices_x_sub_1)
            x_sub_1 =x_sub_1[indices_x_sub_1]  
            
            indices_x_sub_2 = np.arange(len(x_sub_2))
            np.random.shuffle(indices_x_sub_2)
            x_sub_2 =x_sub_2[indices_x_sub_2]  
            
            y_sub_1 =y_sub_1[indices_x_sub_1]          
            y_sub_2 =y_sub_2[indices_x_sub_2]  
            

        else:

            # Shuffle indices
            indices = np.arange(len(x))
            np.random.shuffle(indices)
            
            # Determine the split index
            split_idx = int(len(x) * (1 - split_size))
            if(split_idx==0):
                split_idx += 1
            
            # Split indices
            sub_1_indices = indices[:split_idx]
            sub_2_indices = indices[split_idx:]
            
            # Split the arrays using the indices
            x_sub_1, x_sub_2 = x[sub_1_indices], x[sub_2_indices]
            y_sub_1, y_sub_2 = y[sub_1_indices], y[sub_2_indices]
        
        return x_sub_1, x_sub_2, y_sub_1, y_sub_2
    
    @staticmethod   
    def gen_loader(eeg_signal,eeg_labels,duration=1,bin_dict = None,batchSize=32,psd=True,ar_coef=False,use_simp=True):
        """
            PSD features are extracted from the signal, converted into torch tensors and returned as torch dataloader 

        Args:
            eeg_signal : raw signal 
            eeg_labels : labels in the form of indicies
            duration=1 : percentage of signal to be used
            batchSize=32: batch size of the dataloader

        Returns:
        
            pytorch dataloaders    
        """
        
        features = Utlis.get_psd(eeg_signal,duration=duration,bin_dict=bin_dict,use_simp=use_simp) # getting psd features
        if ar_coef == True:
            # extratcing AR coefs as well
            ar_coefs = Utlis.get_ar_coefs(raw_eeg_signal=eeg_signal,duration=duration)
            features[0] = np.concatenate((features[0],ar_coefs),axis=1)



        features = torch.tensor(features[0])
        labels = torch.tensor(eeg_labels)

        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset,batch_size=batchSize)

        return dataloader

