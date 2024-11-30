from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader,random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import argparse
import time


from networks import Temporal_Encoder , Classification_Head,PositionalEncoding
from packing import ExamplePacking
from dataset import PhysioNet



def train(encoder, classifier, train_loader, val_loader, optimizer_enc, optimizer_cls, criterion, num_epochs, loss_train, loss_val, acc_train, acc_val,patience  = 100 ):

    global best_eval_acc

    tq = tqdm(range(num_epochs), desc="Epochs", unit="epoch")

    
    counter = 0

    for epoch in tq:


        if epoch % 10 == 0 and epoch > 0 :

            x = np.arange(len(train_acc))


            plt.title(f"Accs Until Epoch {epoch}")
            plt.plot(x,acc_train,label="train acc")
            plt.plot(x,acc_val,label="val acc")
            plt.savefig(f"Accs Until Epoch {epoch}.png")

            plt.clf()
            plt.title(f"Losses Until Epoch {epoch}")
            plt.plot(x,loss_train,label="train loss")
            plt.plot(x,loss_val,label="val loss")
            plt.legend()
            plt.savefig(f"Stats Until Epoch {epoch}.png")






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
            optimizer_enc[0].zero_grad()
            optimizer_cls[0].zero_grad()


            # creating attention masks
            # print("packed_batch_eeg.shape ",packed_batch_eeg.shape)


            masks = ExamplePacking.create_attention_mask_batch(batch_identifiers)
            masks = masks.repeat_interleave(encoder.num_heads, dim=0) # duplicating the masks for each head
            masks =masks.to(device)

            # print("masks.shape ",masks.shape)

            # Forward pass through the encoder
            encoded_output = encoder(packed_batch_eeg,masks)

            # unpacking the packed encoded output
            pooled_examples, formatted_labels = ExamplePacking.unpack(packed_sequences=encoded_output,true_lables=batch_labels,identifiers=batch_identifiers)
            formatted_labels = formatted_labels.to(device)

            # Forward pass through the classifier
            outputs = classifier(pooled_examples)

            # Compute loss
            loss = criterion(outputs, formatted_labels)
            epoch_loss += loss.item()

            # Backpropagate
            loss.backward()



            # Step the optimizers

            optimizer_enc[0].step()
            optimizer_cls[0].step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += formatted_labels.size(0)
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

                # print("device ",device)
                masks = masks.to(device)
                encoder.to(device)
                classifier.to(device)
                # Forward pass through the encoder
                encoded_output = encoder(packed_batch_eeg, masks)

                # Unpacking the packed encoded output
                pooled_examples, formatted_labels = ExamplePacking.unpack(packed_sequences=encoded_output, true_lables=batch_labels, identifiers=batch_identifiers)

                formatted_labels = formatted_labels.to(device)

                # Forward pass through the classifier
                outputs = classifier(pooled_examples)

                # Compute validation accuracy and loss
                _, predicted = torch.max(outputs, 1)
                total_val += formatted_labels.size(0)
                correct_val += (predicted == formatted_labels).sum().item()
                val_loss += criterion(outputs, formatted_labels).item()

            val_loss /= len(val_loader)
            val_accuracy = correct_val / total_val

            loss_val.append(val_loss)
            acc_val.append(val_accuracy)

        # scheduler step
        optimizer_enc[1].step()
        optimizer_cls[1].step()


        counter += 1

        stat = {
            'Train Loss': f"{epoch_loss:.4f}",
            'Train Acc': f"{train_accuracy:.2f}",
            'Val Loss': f"{val_loss:.4f}",
            'Val Acc': f"{val_accuracy:.2f}",
            'patience':f'{counter}/{patience}'
        }

        tq.set_postfix(stat)

        with open(logs_path,'a') as f:
            f.write(f"Epoch {epoch} ** stat: {str(stat)}\n" )

        

        # Save best model based on validation accuracy
        if val_accuracy > best_eval_acc:
            # print(f"---- new best val acc achieved {val_accuracy} ----")
            torch.save({'encoder': encoder.state_dict(), 'classifier': classifier.state_dict()}, chk_point_best)
            best_eval_acc = val_accuracy
            counter = 0
            with open(logs_path) as f:
                f.write(f"** Val Acc Increased **\n" )

        torch.save({'encoder': encoder.state_dict(), 'classifier': classifier.state_dict()}, chk_point_last)

        if counter >= patience:
          print("\nearly stopping triggered")
          return loss_train, loss_val, acc_train, acc_val

        # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')

    return loss_train, loss_val, acc_train, acc_val

def sample_windows(x,y,window_size,slide_delta):
    """
    modify the input x,y by extracting the samples based on the window size and the slide_delta
    """

    new_x = []
    new_y = []


    start_idx = 0
    end_idx = start_idx + window_size

    while end_idx <= x.shape[-1]:

        # performing normalization and positional encoding here for convinience
        sub_x = ExamplePacking.standardize_rows(x[:,start_idx:end_idx])
        # print("type(sub_x) ",type(sub_x))
        # print("sub_x.shape ",sub_x.shape)
        # print("torch.tensor(sub_x.T).unsqueeze(0).shape ",torch.tensor(sub_x.T).unsqueeze(0).shape)
        # sub_x = pos_encoder(torch.tensor(sub_x.T).unsqueeze(dim=0))
        # sub_x = sub_x.squeeze().permute(1,0)
        # print("sub_x.shape" ,sub_x.squeeze().shape)

        # sub_x = torch.permute(1,0



        new_x.append(sub_x)
        new_y.append(y)


        start_idx += slide_delta
        end_idx = start_idx + window_size


    return new_x,new_y





def trim_randomly(samples,labels,low=0.4,high=2.01,fixed=None):
    """
    trims the examples randomly by keeping first n points where n is samples from uniform random distribution between [low,high)
    """

    total_samples = samples.shape[0]

    if fixed is None:

      variable_lengths = (np.round(np.random.uniform(low=low,high=high,size=(total_samples)),decimals=1) * 160)
      variable_lengths = variable_lengths.astype(np.int64)

    else:
      variable_lengths = (np.round(np.full(shape=(total_samples),fill_value=fixed),decimals=1) * 160)
      variable_lengths = variable_lengths.astype(np.int64)

    variable_len_covariates = []
    modified_labels = []


    for idx,(sample,labels) in enumerate(zip(samples,labels)):

        x,y = sample_windows(sample,labels,window_size=variable_lengths[idx],slide_delta=16)

        variable_len_covariates += x
        modified_labels += y


    return variable_len_covariates,modified_labels



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


    logs_path = "./logs/log.txt"
    f = open(logs_path,"a")
    f.write("***** ######################################################### *****\n")
    f.write(f"***** Start Time : {time.ctime(time.time())}\n")
    f.close()

    # TODO: complete code  for adding logs for each run you should add new log segment

    num_classes = 106
    num_epochs = 200
    use_scheduler = False
    init_checkpoint_path = None

    parser = argparse.ArgumentParser(description="trains the model")

    parser.add_argument('--lr', type=float, help='Learning rate for the optimizer', required=True)
    parser.add_argument('--epochs', type=float, help='Epochs to train model for', required=False)
    
    parser.add_argument('--classes', type=int, help='Total number of classes', required=False)
    parser.add_argument('--init_checkpoint_path', type=str, help='Path to the checkpoint to load model from', required=False)
    parser.add_argument('--scheduler', type=bool, help='Flag to determine weather to use scheduler or not', required=False)

    
    args = parser.parse_args()

    print("\n####### configuration passed #######")

    print(f"Learning rate: {args.lr}")

    f = open(logs_path,"a")
    f.write(f"args: {vars(args)}\n")
    f.close()

    if args.init_checkpoint_path:
        
        init_checkpoint_path = args.init_checkpoint_path
        
        assert os.path.isfile(init_checkpoint_path), f"no file found at {init_checkpoint_path}"
        print("******* Initialization checkpoint path: ",{args.init_checkpoint_path})
        

    if args.classes:
        num_classes = args.classes
        print("******* Total Number of Classes: ",{args.init_checkpoint_path})

    if args.scheduler:
        use_scheduler = args.scheduler
        print("******* Using Scheduler: ",{args.scheduler})
        
    

    CHECKPOINT_DIR = "./checkpoints"
    chk_point_last = os.path.join(CHECKPOINT_DIR,"last.pth")
    chk_point_best = os.path.join(CHECKPOINT_DIR,"best.pth")
    stats_path = os.path.join(CHECKPOINT_DIR,"stats.pth")
    
    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)



    lr = args.lr
    best_eval_acc = 0.0

    batch_size = int(os.getenv("BATCH_SIZE"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("******* Device: ",{device})
    print("******* Batch Size : ",{batch_size})




    with torch.no_grad():

        pos_encoder = PositionalEncoding(emb_size=64,dropout=0.3,maxlen=320)

        # loading raw_physio net dataset
        # loading raw_physio net dataset
        dataset_unpacked = PhysioNet(activity="all",include_rest=True,sample_windows=False,train=True)
        
        # trimming all examples in the dataset to random lengths
        variable_length_examples,labels = trim_randomly(dataset_unpacked.eeg_raw_x,dataset_unpacked.eeg_data_y.tolist())

        # packing the original examples
        packed_data_tuples = ExamplePacking.pack(samples=variable_length_examples,labels=labels)

        #### creating final datasets and dataloaders ###

        packed_data_tuples = tuple(torch.tensor(np.array(elem)) for elem in packed_data_tuples) # converting to the tensors

        splits = random_split_tensors(0.80,packed_data_tuples[0],packed_data_tuples[1],packed_data_tuples[2])

        packed_dataset_train = TensorDataset(splits[0][0],splits[0][1],splits[0][2])
        packed_loader_trian = DataLoader(dataset=packed_dataset_train,batch_size=batch_size)

        packed_dataset_val = TensorDataset(splits[1][0],splits[1][1],splits[1][2])
        packed_loader_val = DataLoader(dataset=packed_dataset_val,batch_size=batch_size)

    ### ------------------------ ###

    # creating encoder and classifier
   
    eeg_encoder = Temporal_Encoder()
    classifier = Classification_Head(num_classes=num_classes)

    eeg_encoder.to(device)
    classifier.to(device)


    ## ---- loading from checkpoint ---- 

    if init_checkpoint_path is not None:
        
        chk = torch.load(init_checkpoint_path,map_location=device,weights_only=True)
        eeg_encoder.load_state_dict(chk["encoder"])
        classifier.load_state_dict(chk["classifier"])

        print(f"****** loaded weights from {init_checkpoint_path}")
    


    # creating optimizers 
    optimizer_enc = torch.optim.Adam(eeg_encoder.parameters(), lr=lr)
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=lr)
        
    scheduler_enc = torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=10, gamma=0.1)
    scheduler_cls = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=10, gamma=0.1)


    # criterion
    cross_entropy =torch.nn.CrossEntropyLoss()
    
    # stats
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []



    loss_train, loss_val, acc_train, acc_val = train(encoder=eeg_encoder,classifier=classifier,train_loader=packed_loader_trian,val_loader=packed_loader_val,optimizer_enc=(optimizer_enc,scheduler_enc),optimizer_cls=(optimizer_cls,scheduler_cls),criterion=cross_entropy,num_epochs=num_epochs,loss_train=train_loss,loss_val=val_loss,acc_train=train_acc,acc_val=val_acc)
    torch.save({"stats":[loss_train, loss_val, acc_train, acc_val]},stats_path)


    f = open(logs_path,"a")
    f.write(f"***** End Time : {time.ctime(time.time())}\n")
    f.write("***** ######################################################### *****\n")
    f.close()



