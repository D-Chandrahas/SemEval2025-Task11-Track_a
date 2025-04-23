import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datetime import datetime
from os import makedirs


class TextEncoder(torch.nn.Module):
    def __init__(self, model_name, from_pretrained, **kwargs):
        super().__init__()
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if from_pretrained: # pretrained model
            self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False, **kwargs)
        else: # new model
            config = AutoConfig.from_pretrained(model_name, **kwargs)
            self.encoder = AutoModel.from_config(config, add_pooling_layer=False)
        self.config = self.encoder.config

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        return self

    def forward(self, sequences, batch_mode=True, bertviz=False):
        with torch.no_grad():
            if batch_mode: # in batch mode, truncate to max_length and pad to length of longest sequence
                tokenizer_out = self.tokenizer(sequences, return_tensors='pt', padding=True, truncation=True).to(self.device) # tokenize and convert to ids
                input_ids, attention_mask = tokenizer_out.input_ids, tokenizer_out.attention_mask

            else: # in single sequence mode, covert to batch if seq_length > max_length
                if type(sequences) != str: raise TypeError("sequences must be a string when batch_mode is False")

                max_len = self.tokenizer.model_max_length
                tokenizer_out = self.tokenizer(sequences, return_tensors='pt', verbose=False)
                input_ids, attention_mask = tokenizer_out.input_ids, tokenizer_out.attention_mask

                of_len = input_ids.shape[1] - max_len # overflow length
                if of_len > 0: # if seq is longer than max_length, convert to batch of overlapping sequences using sliding window

                    max_seq_len = max_len - 2; overlap_len = 128; stride = max_seq_len - overlap_len
                    config = self.config

                    pad = 0, (stride - (of_len % stride)) % stride # pad to make seq length divisible by stride
                    input_ids = F.pad(input_ids[0, 1:-1], pad, mode="constant", value=config.pad_token_id)

                    temp = torch.empty( ( ( input_ids.shape[0] - max_seq_len) // stride ) + 1, max_seq_len, dtype=torch.int64)
                    for i in range(temp.shape[0]): # reshape to overlapping sequences
                        temp[i] = input_ids[i*stride : i*stride + max_seq_len]

                    temp = F.pad(temp, (1, 0), mode="constant", value=config.bos_token_id) # add beginning of sequence token
                    temp = F.pad(temp, (0, 1), mode="constant", value=config.eos_token_id) # add end of sequence token

                    attention_mask = torch.ones_like(temp)
                    if pad[1] > 0: # if last sequence was padded, set eos to correct position
                        temp[-1, -pad[1]-1] = config.eos_token_id
                        temp[-1, -1] = config.pad_token_id
                        attention_mask[-1, -pad[1]:] = 0 # set attention mask to 0 for padding tokens

                    input_ids = temp

                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask) # encode ids to vectors

        if bertviz: # for bertviz visualization
            return encoder_out.attentions, self.tokenizer.convert_ids_to_tokens(input_ids[0])
        else:
            return encoder_out.last_hidden_state # shape=(batch_size, seq_len, hidden_size)


class ClassificationHead(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.l1 = torch.nn.Linear(input_size, input_size)
        self.l2 = torch.nn.Linear(input_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, output_size)

        self.dropout = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()

        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size
        }

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.dropout(x)
        x = self.relu(self.l2(x))
        x = self.l3(x) # no activation, return logits
        return x # shape=(batch_size, output_size)
    

class EncoderClassifier(torch.nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.device = "cpu"
        self.encoder = encoder.eval()
        self.classifier = classifier.eval()

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        self.classifier.to(device)
        return self

    def forward(self, sequences, batch_mode=True):
        return self.classifier(self.encoder(sequences, batch_mode)[:,0]) # use first token (CLS token) as representation for classification
    
    def __call__(self, sequences, batch_mode=True):
        return self.forward(sequences, batch_mode)

    def fit(self, optimizer, optimizer_kwargs, loss_cls, loss_cls_kwargs, train_loaders, valid_loaders, epochs, loss_target=0.0, save_path="./ckpts/", resume_from=None):
        if save_path and save_path[-1] != "/" and save_path[-1] != "\\": save_path += "/"
        makedirs(save_path, exist_ok=True)

        if isinstance(train_loaders, torch.utils.data.DataLoader): train_loaders = (train_loaders,) # if only one dataloader is passed, convert to singleton tuple
        if isinstance(valid_loaders, torch.utils.data.DataLoader): valid_loaders = (valid_loaders,)

        optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        criterion = loss_cls(**loss_cls_kwargs)
        start_epoch = 1

        if resume_from: # resume training from checkpoint
            epoch, optimizer_state_dict = self.load_ckpt(resume_from)
            start_epoch = epoch + 1
            optimizer.load_state_dict(optimizer_state_dict)

        num_batches_train = list(map(len, train_loaders)) # number of batches in each dataloader
        num_batches_valid = list(map(len, valid_loaders))

        for epoch in range(start_epoch, epochs+1):

            self.train()
            train_loss = 0.0

            curr_batch_list = [0 for _ in num_batches_train] # progress for each dataloader

            for i_dataloader, dataloader in enumerate(train_loaders): # iterate over dataloaders
                n_batches = num_batches_train[i_dataloader]
                try:
                    loss_mask = dataloader.loss_mask
                except AttributeError:
                    loss_mask = slice(None)

                for i_batch, (sequences, labels) in enumerate(dataloader): # iterate over batches
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    pred_labels = self.forward(sequences)
                    loss = criterion(pred_labels[:, loss_mask], labels[:, loss_mask]) # compute loss only for columns/labels present in curr dataset
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    if i_batch % (n_batches//100 + 1) == 0: # update progress bar every after every ~1% of batches
                        curr_batch_list[i_dataloader] = i_batch + 1
                        progress_bar = ""
                        for curr_batch, num_batches in zip(curr_batch_list, num_batches_train):
                            progress_bar += f"{curr_batch}/{num_batches}, "
                        print(f"Epoch {epoch}/{epochs} - Training : {progress_bar[:-2]}", end="\r", flush=True)
            train_loss /= sum(num_batches_train) # average training loss for epoch
            print(" " * 100, end="\r")

            with torch.inference_mode(): # validation

                self.eval()
                valid_loss = 0.0

                curr_batch_list = [0 for _ in num_batches_valid]

                for i_dataloader, dataloader in enumerate(valid_loaders):
                    n_batches = num_batches_valid[i_dataloader]
                    try:
                        loss_mask = dataloader.loss_mask
                    except AttributeError:
                        loss_mask = slice(None)

                    for i_batch, (sequences, labels) in enumerate(dataloader):
                        labels = labels.to(self.device)

                        pred_labels = self.forward(sequences)
                        loss = criterion(pred_labels[:, loss_mask], labels[:, loss_mask])

                        valid_loss += loss.item()

                        if i_batch % (n_batches//100 + 1) == 0:
                            curr_batch_list[i_dataloader] = i_batch + 1
                            progress_bar = ""
                            for curr_batch, num_batches in zip(curr_batch_list, num_batches_valid):
                                progress_bar += f"{curr_batch}/{num_batches}, "
                            print(f"Epoch {epoch}/{epochs} - Validation : {progress_bar[:-2]}", end="\r", flush=True)
                valid_loss /= sum(num_batches_valid)
                print(" " * 100, end="\r")

            self.save_ckpt(epoch, optimizer.state_dict(), f"{save_path}model_{epoch}_{datetime.now().strftime('%H%M%S')}.ckpt") # save checkpoint
            print(f"Epoch {epoch}/{epochs} - Train loss: {train_loss:.6f}, Valid loss: {valid_loss:.6f}") # print training and validation loss for epoch
            if valid_loss <= loss_target: break # early stopping


    def save_ckpt(self, epoch, optimizer_state_dict, path):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer_state_dict
        }, path)

    def load_ckpt(self, path):
        ckpt = torch.load(path, weights_only=True)
        self.load_state_dict(ckpt["model_state_dict"])
        self.train()
        return ckpt["epoch"], ckpt["optimizer_state_dict"]

    def save(self, path, comment=None): # save model architecture and weights
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": {
                "encoder": self.encoder.config._name_or_path,
                "classifier": self.classifier.config
            },
            "comment": comment
        }, path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"])
        self.eval()
    
    @classmethod
    def from_trained(cls, path, print_comment=False, **kwargs): # construct model from model save file
        arch = torch.load(path, weights_only=True)
        model = cls(TextEncoder(arch["config"]["encoder"], False, **kwargs),
                    ClassificationHead(**arch["config"]["classifier"]))
        model.load_state_dict(arch["model_state_dict"])
        if print_comment: print(arch["comment"])
        return model.eval()
