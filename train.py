from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader,random_split
from sklearn.model_selection import train_test_split


from networks import Temporal_Encoder , Classification_Head
from packing import ExamplePacking
from dataset import PhysioNet







def train(encoder, classifier, train_loader, val_loader, optimizer_enc, optimizer_cls, criterion, num_epochs, loss_train, loss_val, acc_train, acc_val):

    global best_eval_acc
    
    tq = tqdm(range(num_epochs), desc="Epochs", unit="epoch")

    for epoch in tq:
        encoder.train()
        classifier.train()

        epoch_loss = 0
        correct_train = 0
        total_train = 0

        # Training loop
        for batch_idx, (packed_batch_eeg, batch_labels, batch_identifiers) in enumerate(train_loader, start=1):
            
            # Update the tqdm description to show the current batch number
            tq.set_description(f"Train Epoch {epoch+1}/{num_epochs}, it {batch_idx}/{len(train_loader)}")
         
            # Ensure correct shape of batch_data
            packed_batch_eeg = packed_batch_eeg.permute(0, 2, 1)  # Assuming the input needs permuting to [batch_size, channels, length]
            packed_batch_eeg = packed_batch_eeg.float().to(device)
            batch_labels = batch_labels.long().to(device)


            # Zero the gradients for both encoder and classifier optimizers
            optimizer_enc.zero_grad()
            optimizer_cls.zero_grad()


            # creating attention masks
            # print("packed_batch_eeg.shape ",packed_batch_eeg.shape)

            
            masks = ExamplePacking.create_attention_mask_batch(batch_identifiers)
            masks = masks.repeat_interleave(encoder.num_heads, dim=0) # duplicating the masks for each head

            # print("masks.shape ",masks.shape)

            # Forward pass through the encoder
            encoded_output = encoder(packed_batch_eeg,masks)
            
            # unpacking the packed encoded output
            pooled_examples, formatted_labels = ExamplePacking.unpack(packed_sequences=encoded_output,true_lables=batch_labels,identifiers=batch_identifiers)


            # Forward pass through the classifier
            outputs = classifier(pooled_examples)

            # Compute loss
            loss = criterion(outputs, formatted_labels)
            epoch_loss += loss.item()

            # Backpropagate
            loss.backward()

            # Step the optimizers
            optimizer_enc.step()
            optimizer_cls.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == formatted_labels).sum().item()

        # Calculate and save the epoch loss and accuracy for training
        epoch_loss /= len(train_loader)
        train_accuracy = correct_train / total_train
        loss_train.append(epoch_loss)
        acc_train.append(train_accuracy)

        # Validation loop
        encoder.eval()
        classifier.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            for batch_idx, (packed_batch_eeg, batch_labels, batch_identifiers) in enumerate(val_loader, start=1):
            
            # Update the tqdm description to show the current batch number
                tq.set_description(f"Validation Epoch {epoch+1}/{num_epochs}, it {batch_idx}/{len(train_loader)}")

                # Ensure correct shape of batch_data
                packed_batch_eeg = packed_batch_eeg.permute(0, 2, 1)
                packed_batch_eeg = packed_batch_eeg.float().to(device)
                batch_labels = batch_labels.long().to(device)

                # Creating attention masks
                masks = ExamplePacking.create_attention_mask_batch(batch_identifiers)
                masks = masks.repeat_interleave(encoder.num_heads, dim=0)  # Duplicating the masks for each head

                # Forward pass through the encoder
                encoded_output = encoder(packed_batch_eeg, masks)

                # Unpacking the packed encoded output
                pooled_examples, formatted_labels = ExamplePacking.unpack(packed_sequences=encoded_output, true_lables=batch_labels, identifiers=batch_identifiers)

                # Forward pass through the classifier
                outputs = classifier(pooled_examples)

                # Compute validation accuracy and loss
                _, predicted = torch.max(outputs, 1)
                total_val += batch_labels.size(0)
                correct_val += (predicted == formatted_labels).sum().item()
                val_loss += criterion(outputs, formatted_labels).item()

            val_loss /= len(val_loader)
            val_accuracy = correct_val / total_val

            loss_val.append(val_loss)
            acc_val.append(val_accuracy)

        # Save best model based on validation accuracy
        if val_accuracy > best_eval_acc:
            print(f"---- new best val acc achieved {val_accuracy} ----")
            torch.save({'encoder': encoder.state_dict(), 'classifier': classifier.state_dict()}, chk_point_best)
            best_eval_acc = val_accuracy

        torch.save({'encoder': encoder.state_dict(), 'classifier': classifier.state_dict()}, chk_point_last)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')

    return loss_train, loss_val, acc_train, acc_val


def trim_randomly(samples,low=0.4,high=2.01):
    """
    trims the examples randomly by keeping first n points where n is samples from uniform random distribution between [low,high)
    """

    total_samples = samples.shape[0] 
    variable_lengths = (np.round(np.random.uniform(low=low,high=high,size=(total_samples)),decimals=1) * 160)
    variable_lengths = variable_lengths.astype(np.int64)
    
    variable_len_covariates = []

    for idx,sample in enumerate(samples):    
        variable_len_covariates.append(sample[:,:variable_lengths[idx]])

    
    return variable_len_covariates


### -- thanks to ChatGPT --- ###

def random_split_tensors(split_ratio, *tensors, seed=None):
    """
    Split tensors into train and validation sets with the same element-to-element correspondence.
    
    Args:
        split_ratio (float): Proportion of data to be used for training (e.g., 0.8 for 80% training).
        *tensors (torch.Tensor): Any number of tensors to split (must have the same first dimension).
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        Tuple of tuples: Each tuple contains the train and validation split of a tensor.
    """
    assert all(tensor.size(0) == tensors[0].size(0) for tensor in tensors), "All tensors must have the same first dimension."
    # for elem in tensors:
    #     print("type(elem) ",type(elem))

    if seed is not None:
        torch.manual_seed(seed)
    
    indices = list(range(tensors[0].size(0)))
    
    # Split the indices for train and validation based on split_ratio
    train_indices, val_indices = train_test_split(indices, train_size=split_ratio, random_state=seed)
    
    # Create train and validation splits for each tensor
    split1 = []
    split2 = []

    for tensor in tensors:
        train_split = tensor[train_indices]
        val_split = tensor[val_indices]
        
        split1.append(train_split)
        split2.append(val_split)
    
    return tuple((split1,split2))



if __name__ == "__main__":
    best_eval_acc = 0
    chk_point_best = "./best_chk.pth"
    chk_point_last = "./last_chk.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # loading raw_physio net dataset
    dataset_unpacked = PhysioNet(activity="all",include_rest=True,sample_windows=False)

    # trimming all examples in the dataset to random lengths
    variable_length_examples = trim_randomly(dataset_unpacked.eeg_raw_x)

    # packing the original examples 
    packed_data_tuples = ExamplePacking.pack(samples=variable_length_examples,labels=dataset_unpacked.eeg_data_y.tolist())
    
    #### creating final datasets and dataloaders ###

    packed_data_tuples = tuple(torch.tensor(np.array(elem)) for elem in packed_data_tuples) # converting to the tensors
    
    splits = random_split_tensors(0.80,packed_data_tuples[0],packed_data_tuples[1],packed_data_tuples[2])

    # for split in splits:
    #     for tensor in split:
    #         print("tensor.shape ",tensor.shape)

    #     print("*#*")    
    
    packed_dataset_train = TensorDataset(splits[0][0],splits[0][1],splits[0][2])
    packed_loader_trian = DataLoader(dataset=packed_dataset_train,batch_size=4)

    packed_dataset_val = TensorDataset(splits[1][0],splits[1][1],splits[1][2])
    packed_loader_val = DataLoader(dataset=packed_dataset_val,batch_size=4)

    ### ------------------------ ###

    # creating encoder and classifier
    eeg_encoder = Temporal_Encoder()
    classifier = Classification_Head(num_classes=10)
    
    lr = 0.001

    # creating optimizers 
    optimizer_enc = torch.optim.Adam(eeg_encoder.parameters(), lr=lr)
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=lr)

    # criterion
    cross_entropy =torch.nn.CrossEntropyLoss()
    
    # stats
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []



    outs = train(encoder=eeg_encoder,classifier=classifier,train_loader=packed_loader_trian,val_loader=packed_loader_val,optimizer_enc=optimizer_enc,optimizer_cls=optimizer_cls,criterion=cross_entropy,num_epochs=10,loss_train=train_loss,loss_val=val_loss,acc_train=train_acc,acc_val=val_acc)
    

    ### test the maxpooling funciotn in packing and modify it to return a tensor not a list





