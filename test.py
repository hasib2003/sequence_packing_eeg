"""
used for testing the model on variable length inputs
"""


from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader,random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

from networks import Temporal_Encoder , Classification_Head,PositionalEncoding
from packing import ExamplePacking
from dataset import PhysioNet







def test(encoder, classifier, test_loader, criterion):

    encoder.eval()
    classifier.eval()

    with torch.no_grad():

        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        for batch_idx, (packed_batch_eeg, batch_labels) in enumerate(test_loader, start=1):
        
        # Update the tqdm description to show the current batch number
            # tq.set_description(f"Testing iter {batch_idx}/{len(test_loader)}")

            # Ensure correct shape of batch_data
            packed_batch_eeg = packed_batch_eeg.permute(0, 2, 1)
            packed_batch_eeg = packed_batch_eeg.float().to(device)
            batch_labels = batch_labels.long().to(device)

            # Forward pass through the encoder            
            encoded_output = encoder(packed_batch_eeg)
            pooled = torch.max(encoded_output,dim=1).values
            
            # print("packed_batch_eeg.shape ",packed_batch_eeg.shape)
            # print("encoded_output.shape ",encoded_output.shape)
            # print("pooled.shape ",pooled.shape)

            # # Forward pass through the classifier
            outputs = classifier(pooled)

            # print("outputs.shape ",outputs.shape)
            # print("batch_labels.shape ",batch_labels.shape)

            # Compute validation accuracy and loss
            _, predicted = torch.max(outputs, 1)
            total_val += batch_labels.size(0)
            correct_val += (predicted == batch_labels).sum().item()
            val_loss += criterion(outputs, batch_labels).item()

        val_loss /= len(test_loader)
        val_accuracy = correct_val / total_val

    return val_loss,val_accuracy


def gen_loader(eeg_x,eeg_y,signal_length=0.4,batch_size=16):

    assert 0.4 <= signal_length and signal_length <=2.0 , "signal length must be between 0.4 and 2.0"
    # print("eeg_x.shape ",eeg_x.shape)
    
    sub_eeg = torch.tensor(eeg_x[:,:,:int(signal_length*160)])
    eeg_y = torch.tensor(eeg_y).reshape(-1)

    # print("eeg_y.shape ",eeg_y.shape)
    
    # print("sub_eeg.shape ",sub_eeg.shape)


    dataset = TensorDataset(sub_eeg,eeg_y)
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size)

    return dataloader
    


    



if __name__ == "__main__":

    chk_point_best = os.getenv("BEST_CHECKPOINT")
    # chk_point_last = "/content/drive/MyDrive/last_chk.pth"

    batch_size = int(os.getenv("BATCH_SIZE")) 
    device = "cuda" if torch.cuda.is_available() else "cpu"



    dataset_unpacked = PhysioNet(activity="all",include_rest=True,sample_windows=False,train=False)

    signal_len = 0.5

    test_loader = gen_loader(dataset_unpacked.eeg_raw_x,dataset_unpacked.eeg_data_y,signal_length=signal_len)

    ### ------------------------ ###

    # creating encoder and classifier
    eeg_encoder = Temporal_Encoder()
    classifier = Classification_Head(num_classes=106)


    # loading from chkpoint
    chk_pnt = torch.load(chk_point_best,map_location=torch.device(device))


    eeg_encoder.load_state_dict(chk_pnt["encoder"])
    classifier.load_state_dict(chk_pnt["classifier"])

    print(f" ###### Chkpoint Loaded From {chk_point_best} ###### ")

    

    cross_entropy =torch.nn.CrossEntropyLoss()

    test_loss,test_acc = test(encoder=eeg_encoder,classifier=classifier,test_loader=test_loader,criterion=cross_entropy)
    
    print(f" ###### signal_len {signal_len} test_loss {test_loss} test_acc {test_acc} ###### ")








