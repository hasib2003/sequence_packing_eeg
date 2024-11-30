
import numpy as np
import torch

class ExamplePacking:
    
    def __init__(self,max_seq=4*160,max_search_depth=17) -> None:

        super().__init__()

        """
        max_seq(int):  maximum sequence length
        max_search_depth(int): determines how long should it search for best fit before padding            
        """

        self.max_bin_size = max_seq
        self.max_search_depth = max_search_depth
        


        # self.packed_arr1,self.packed_arr2,self.packing_identifiers =  self.pack(arr1.coppacked_sequences(),arr2.coppacked_sequences())


    @staticmethod
    def standardize_rows(arr):
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

    @staticmethod
    def pack(samples,labels,max_bin_size=3*160,max_search_depth=17):

        """pack the given list of samples according to max_seq length and depth search args in the constructor
        
        samples: must be of shape (seqlen,*)
        
        """

        packing_identifiers = [] # used to trace back the position and length of original sequences in a packed sequence
        bins = [] # contains the packed sequences
        bin_labels = [] # contains the respective labels of the packed sequences


        while len(samples) > max_search_depth:
            
            #### filling a new bin

            bin_used = 0
            depth_explored = 0


            identifier = np.full(shape=(max_bin_size),fill_value=-1).astype(np.float16)
            # print("identifier.shape ",identifier.shape)
            
            tar_bin = np.full(shape=(samples[0].shape[0],max_bin_size),fill_value=-1).astype(np.float32)
            tar_labels = np.full(shape=(max_bin_size,),fill_value=-1)

            examples_added = 0 # counter for no of examples packed


            while bin_used < max_bin_size and depth_explored <= max_search_depth and len(samples) > max_search_depth: 
                # print("len(samples) ",len(samples))
                # print("depth_explored",depth_explored)
                
                available_bin = max_bin_size - bin_used

                curr_seq_length = samples[depth_explored].shape[-1]
                # adding the sample into bin if it does not exceed the max size

                if curr_seq_length <= available_bin :

                    #### adding one sample to a bin
                                        
                    # print("samples[depth_explored] ",samples[depth_explored])
                    # print("before tar_bin ",tar_bin[:,bin_used:bin_used+curr_seq_length])

                    tar_bin[:,bin_used:bin_used+curr_seq_length] = samples[depth_explored]
                    
                    # print("tar_bin afters ",tar_bin[:,bin_used:bin_used+curr_seq_length])
                    # print(f"Sample shape: {samples[depth_explored].shape}, Bin shape: {tar_bin[:, bin_used:bin_used + curr_seq_length].shape}")


                    tar_labels[bin_used:bin_used+curr_seq_length] = labels[depth_explored]

                    samples.pop(depth_explored)
                    labels.pop(depth_explored)

                    # print(label)

                    identifier[bin_used:bin_used+curr_seq_length] = examples_added

                    examples_added += 1

                    depth_explored = 0
                    bin_used  += curr_seq_length

                else:

                    # if sample can not be added incrementing the depth_explored
                    depth_explored += 1
            

            packing_identifiers.append(identifier)
            bins.append(tar_bin)
            bin_labels.append(tar_labels)

        return bins,bin_labels,packing_identifiers
    
    @staticmethod
    def unpack(packed_sequences,true_lables,identifiers,max_pool=True):
        """  
        unpacks the sequence 

        returns 
            unpacked sequences, lables  if max_pool false            
            unpacked sequences max_pooled along dim=0 , lables  if max_pool true

        """

        # print("input sequence packed ",packed_sequences.shape)


        # output_tensor = torch.zeros(size=(packed_sequences.shape[0],63))

        output_tensor = None
        
        unpacked_predictions = []    
        unpacked_lables = []

        max_pooled = []

        for packed_sequence,seq_iden,labels in zip(packed_sequences,identifiers,true_lables):
            
            seq_nums , locations= np.unique(seq_iden,return_index=True) 
            locations = locations.tolist() 

            # print("seq_nums ",seq_nums)
            if seq_nums[0] == -1: # if there is anpacked_sequences padding
                # print("padding found")
                padding_start_loc = locations.pop(0)

                pass
            else:
                # print("seq_iden.shape ",seq_iden.shape)
                padding_start_loc = seq_iden.shape[0]

            locations.append(padding_start_loc)


            ## unpacking the sequence 
            # print("locations ",locations)
            for idx in range(len(locations)-1):
                example_unpacked = packed_sequence[locations[idx]:locations[idx+1],:]

                # print("example_unpacked ",example_unpacked.shape)

                unpacked_predictions.append(example_unpacked)

                if max_pool == True:

                    ten = example_unpacked.clone().detach().requires_grad_(True)
                    # print("example_unpacked ",example_unpacked.shape)
                    # print("ten.shape ",ten.shape)
                    axis_pooled = torch.max(ten,dim=0)
                    
                    vals = axis_pooled.values.reshape(1,-1)

                    max_pooled.append(vals)
                    
                    if output_tensor is not None:
                        output_tensor = torch.cat((output_tensor,vals),dim=0)
                    else:
                        output_tensor = vals


                    # print("axis_pooled.values.shape ",axis_pooled.values.shape)

            ## unpacking the labels correspoding to the unpacked sequences
            unpacked_lables.extend(labels[locations[:-1]].tolist())

        


        if max_pool == True:
            # print("max_pooled ",max_pooled)
            # print("output_tensor.shape ",output_tensor.shape)
            return output_tensor,torch.tensor(unpacked_lables)

        return torch.tensor(np.array(unpacked_predictions)),torch.tensor(unpacked_lables)

    def create_attention_mask_batch(batch_example_list):
        """
        Create an attention mask for a batch of sequences where different examples 
        within each sequence do not attend to each other.
        
        Args:
        - batch_example_list (list of lists or 2D tensor): A batch of sequences 
        representing the examples, e.g., [[1, 1, 1, 2, 2], [1, 2, 2, 3, 3]]
        
        Returns:
        - attention_mask (torch.Tensor): A 3D tensor (batch_size, seq_len, seq_len), 
                                        where 1 means positions can attend to each other,
                                        and 0 means they cannot.
        """
        # Convert the input list to a 2D tensor (batch_size, seq_len)
        batch_example_list = np.array(batch_example_list)
        sequence_tensor = torch.tensor(batch_example_list)
        
        # Create a comparison matrix for each batch: (batch_size, seq_len, seq_len)
        attention_mask = (sequence_tensor.unsqueeze(1) == sequence_tensor.unsqueeze(2))
        
        return attention_mask
