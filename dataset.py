"""Loads dataset required by the model"""

# working with subsets of files for one subject is not supported yet

from torch.utils.data import Dataset
import os
import numpy as np
import mne
# custom
import config
from pre_processors import PreProcessor
from utlis import Utlis
import random

class PhysioNet(Dataset):

    def __init__(self,activity="all",window_length=0.5,slide_delta=0.1,sample_windows=False,include_rest=False,extract_delta=False,train=True,k=3) -> None:
        super().__init__()


        self.activity_name = activity
        self.activity_list = ["eye_open","eye_closed","fist_real","fist_imagined","both_real","both_imagined","all"]

        """
        Args:
            activity: determines which activity is to be used for authenticaion prupose
                        must be one of the following 
                            1. fist_real -> subject opens and closes the corresponding fist until the target disappears.
                            2. fist_imagined -> The subject imagines opening and closing the corresponding fist until the target disappears
                            3. both_real -> The subject opens and closes either both fists (if the target is on top) or both both (if the target is on the bottom) until the target disappears.
                            4. both_imagined -> The subject imagines opening and closing either both fists (if the target is on top) or both both (if the target is on the bottom) until the target disappears
                            5. eye_open -> baseline task
                            6. eye_closed -> baseline task
                            7. all

            sample_windows : determines weather to make overlapping windows or not, if false only first window_length from each trail is used
            include_rest : determines weather to use resting state signal in between the samples or not
            include_rest : determines weather to use resting state signal in between the samples or not
            window_length : determines how many seconds are contained in one sample
            slide_delta: determines the difference between two consecutive windows in seconds
            extract_delta: determines weather to filter delta_band or not (0.5hz - 3.5hz)
            train : loads data from first 12 epochs if true, else loads from 12-15 epochs
        """


        assert activity in self.activity_list, f"activity {activity} not in {self.activity_list}"
        self.root_dir = os.getenv('PATH_ROOT_DIR')  # root dir containing all files
        self.path_records = os.getenv('PATH_RECORDS') # file containg names of all .edf files
        self.files_per_subj = int(os.getenv('FILES_PER_SUBJ')) # file containg names of all .edf files
        self.extract_delta = extract_delta


        self.num_subjs =  int(os.getenv("NUM_SUBJS"))  # total number of subjs

        assert self.files_per_subj <= 14, f"Each subject can have max 14 files"
        assert self.num_subjs <=109 , f"Max 109 subjects are allowed" 

        assert os.path.isdir(self.root_dir) , f"No directory found at {self.root_dir}"
        assert os.path.isfile(self.path_records), f"No file found at {self.path_records}"





        self.k = k
        self.include_rest = include_rest # determines weather to include rest signal in between trails or not
        self.sample_rate = 160
        self.window_length = window_length
        self.slide_delta = slide_delta
        self.isTrain = train 

        self.window_size = int(self.sample_rate * self.window_length) # number of points to be used in one window
        self.slide_delta_size = int(self.sample_rate * self.slide_delta)

        
        print(f"======== dataset configuration (mode) {'training' if self.isTrain else 'testing'} ========\n")

        print(f"======== k -> {self.k} ========")
        print(f"======== sample_windows -> {sample_windows} ========")
        print(f"======== include_rest -> {self.include_rest} ========")
        print(f"======== include_rest -> {self.include_rest} ========")
        print(f"======== extract_delta -> {self.extract_delta} ========")
        print(f"======== activity_name -> {self.activity_name} ========")
        print(f"======== window_length -> {self.window_length} ========")
        print(f"======== slide_delta -> {self.slide_delta} ========\n")
        print(f"======== .......... ========")


        self.eeg_raw_x = None
        self.eeg_data_y = None

        self.eeg_raw_windowed = None
        self.eeg_y_windowed = None

        self.load_meta() # loadin meta data i.e. file names
        self.load_mem() # loading actual epoched data from edf files based on the activity name provided

        if sample_windows == True:
            self.sample_windows() # creating windows out of epochs
        else:
            self.eeg_raw_windowed = self.eeg_raw_x[:,:,:self.window_size]
            self.eeg_y_windowed = self.eeg_data_y

            

        self.total_samples = self.eeg_raw_windowed.shape[0]   
        # self.total_samples = self.eeg_raw_x.shape[0] * 4


        self.samples_per_subject = 180 #, 3 * 15 * 4 files_per_subj*sample_per_file*windows_per_sample

        if self.include_rest == True:
            self.samples_per_subject = 180 * 2 # because there are 15 rest and 15 activity trials for each subject in each file

        # self.eeg_raw_x = np.transpose(self.eeg_raw_x,(1,0))
    
    def sample_windows(self):
        """  
        modify the eeg_raw_x by extracting the samples based on the window size and the slide_delta        
        """

        new_samples_num = self.eeg_raw_x.shape[0] * ( 1 + (self.eeg_raw_x.shape[-1] - self.window_size)//self.slide_delta_size)

        modified_data = np.zeros(shape=(new_samples_num,self.eeg_raw_x.shape[1],self.window_size))
        modified_labels = []

        run_idx = 0

        for sample,cls in zip(self.eeg_raw_x,self.eeg_data_y):
            
            start_idx = 0
            end_idx = self.window_size 
            
            while end_idx <= sample.shape[-1]:
                
                modified_data[run_idx] = sample[:,start_idx:end_idx]

                run_idx +=1 
                start_idx += self.slide_delta_size
                end_idx  += self.slide_delta_size
                modified_labels.append(cls)

        self.eeg_raw_windowed = modified_data
        self.eeg_y_windowed = modified_labels

        assert self.eeg_raw_windowed.shape[0] == len(self.eeg_y_windowed), "num of samples must match the number of labels"


    def filter_paths(self,file_name):
        """
        filters the paths according to the activity required by the user
            filtering assumes following structure
                file1 -> baseline, eyes open
                file2 -> baseline, eyes close
                file3 -> Task 1 (open and close left or right fist)
                file4 -> Task 2 (imagine opening and closing left or right fist)
                file5 -> Task 3 (open and close both fists or both both)
                file5 -> Task 4 (imagine opening and closing both fists or both both)
                .
                .
                .        
        """

        idx = self.activity_list.index(self.activity_name) + 1
        
        if idx == len(self.activity_list) : # if activity_name is "all"  always returning trye
            return True 


        file_num = int(file_name.split("R")[-1].split(".")[-2])
        if idx > 2: 
            # if task is not eyes open or close
            if file_num >= idx and (file_num - idx ) % 4 == 0:
                return True 
            return False

        else:
            return idx == file_num

    def load_meta(self):

        """
        searches for the paths of the edf files and load them into mem
        """

        file_names = []

        with open(self.path_records,"r") as f:
            
            content = f.readline()
            while(content):
                file_name = content.strip()
                # # checking if the file belongs to the given activity
                if self.filter_paths(file_name) == False:
                    content = f.readline()  
                    continue
                
                file_names.append(file_name)
                content = f.readline()


        # print(f"following records found {file_names}")

        assert len(file_names) >= self.num_subjs * self.files_per_subj , f"Expected {self.num_subjs * self.files_per_subj} or more files. Found {len(file_names)}"
        # allowing more files to be able to work with subsets of data as well

        self.files = [] # list containg the paths of all edf. files

        for file_name in file_names:
            self.files.append(os.path.join(self.root_dir,file_name))
        
        
        # print(f"files with path {self.files}")
        print(f".... found {len(self.files)} edf files ....")

    def load_mem(self):
        """
        loads data into the memory in form of np arrays
        """

        excluded = []


        for id in range(self.num_subjs):

            if id+1 in [88,92,100]: # skipping these subjects since they have sampling rate of 128 hertz instead of 160 hertz
                print(f"---- data from subject {id+1} is being excluded because of lesser sampling rate ---- ")
                excluded.append(id+1)
                continue

            subj_data = self.load_subj(id+1)        

            fill_value = id
            if len(excluded) > 0:
                fill_value = id - len(excluded) 


            subj_ids = np.full((subj_data.shape[0],1),fill_value=fill_value)

            if(self.eeg_raw_x is not None):
                self.eeg_raw_x = np.concatenate((self.eeg_raw_x,subj_data),axis=0)
                self.eeg_data_y = np.concatenate((self.eeg_data_y,subj_ids),axis=0)

            else:
                self.eeg_raw_x = subj_data
                self.eeg_data_y = subj_ids
        
        self.num_subjs -= len(excluded)
        self.excluded = excluded
        print(f"---- data loaded from total of {self.num_subjs} -----")
        
    
    def load_subj(self,subj_id):
        """
        given the subject id i.e. 1,2,3 ...
        returns the all of epochs of that subject
        """
           

        start_idx = (subj_id - 1) * self.files_per_subj
        end_idx = start_idx + self.files_per_subj

        subj_files = self.files[start_idx:end_idx] # contains files related to concerned subject only         
        # print(f"subj with id {subj_id} files are ",subj_files)
        subj_data = PreProcessor.get_epochs(subj_files,inlcude_rest=self.include_rest,extract_delta=self.extract_delta,isTrain=self.isTrain,k_val=self.k)
        # print("subj_data.shape ",subj_data.shape)
        

        return subj_data
    
    def standardize_rows(self,arr):
        """
        Standardize each row of the array to have a mean of 0 and a standard deviation of 1.
        
        Parameters:
        arr (numpy.ndarray): Input array of shape (64, 80)
        
        Returns:
        numpy.ndarray: Standardized array with the same shape as input
        """
        # Calculate the mean and standard deviation for each row
        row_mean = arr.mean(axis=1, keepdims=True)
        row_std = arr.std(axis=1, keepdims=True)
        
        # Standardize each row
        standardized_arr = (arr - row_mean) / row_std
        
        return standardized_arr

    def __len__(self):

        return self.total_samples
    
    def get_sample(self,sample_idx,window_idx):
        #  print(f"got sapmle_idx {sample_idx} window_idx {window_idx}")
         return self.eeg_raw_x[sample_idx][:,window_idx*self.window_size:(window_idx+1)*self.window_size]

    def __getitem__(self, index) :
        
        raw = self.eeg_raw_windowed[index]
        standardized = self.standardize_rows(raw)

        psds = self.get_psd(raw) # extracting the power spectral density features 
        psds = psds[:,:32]
        label =  self.eeg_y_windowed[index][0]

        # print("type(psds),type(standardized) ",type(psds),type(standardized))


        return np.transpose(standardized), psds, label

    def get_psd(self,raw_eeg_signal):
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
        


        best_fft = 64 # no of points to be used in calculation of fast fourier transform
        curr_fft= best_fft



        # features_list = [] # list containing all of the psd values for each sample

        
        psds, freqs = mne.time_frequency.psd_array_welch(raw_eeg_signal, fmin=0, fmax=np.inf,sfreq=160, n_fft=curr_fft)

        return  psds




if __name__ == "__main__":
    data = PhysioNet(activity="fist_real",include_rest=False,window_length=0.5,slide_delta=0.5,extract_delta=True,train=False,k=1)

    
    print(f"--total samples {data.__len__()} --")
    for idx,i in enumerate(data):
        if idx == 0:
            print(i[0])
        pass
    print("data scanned successfully")