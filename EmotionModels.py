import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from BaseModels import TextEncoder, ClassificationHead, EncoderClassifier
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from collections import defaultdict


class EmotionDataset(Dataset):
    cols = np.array(("text", "anger", "disgust", "fear", "joy", "sadness", "surprise"))
    labels = cols[1:]
    def __init__(self, path):
        dtypes = defaultdict(lambda: str, ((label, np.float32) for label in self.labels)) # set dtypes for columns
        self.data = pd.read_csv(path, header=0, dtype=dtypes)
        self.data = self.data.reindex(columns=self.cols, fill_value=np.float32(-1)) # reorder columns according to {cols} & fill missing labels with -1
        self.loss_mask = [(x >= 0) for x in self.data.iloc[0, 1:]] # boolean list indicating non-missing labels
        self.labels = self.__class__.labels[self.loss_mask]
        self.lang = path[-7:-4]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index, 0], torch.tensor(self.data.iloc[index, 1:].array, dtype=torch.float32) # returns text and labels as tensors
    

class EmotionDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1):
        self.loss_mask = dataset.loss_mask
        self.labels = dataset.labels
        self.lang = dataset.lang
        super().__init__(dataset, batch_size)


class EmotionClassifier(torch.nn.Module):
    def __init__(self, model_name=None, from_pretrained=False, **kwargs):
        super().__init__()
        self.device = "cpu"
        if "path" in kwargs: # load trained weights
            self.model = EncoderClassifier.from_trained(**kwargs)
        else: # untrained/pretrained model
            encoder = TextEncoder(model_name, from_pretrained=from_pretrained, **kwargs)
            classification_head = ClassificationHead(encoder.config.hidden_size, 384, len(EmotionDataset.labels))
            self.model = EncoderClassifier(encoder, classification_head)

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self
    
    def evaluate(self, data):
        if type(data) != list and type(data) != tuple:
            data = (data,)
        self.eval()
        with torch.inference_mode():
            for dataloader in data:
                y_true, y_pred = [], [] # lists to store true and predicted labels for each batch
                loss_mask = dataloader.loss_mask
                labels = dataloader.labels

                for sequences, true_labels in dataloader:
                    y_true.append((true_labels[:, loss_mask]).numpy(force=True))
                    pred_labels = self.model(sequences)
                    y_pred.append((pred_labels[:, loss_mask] > 0).numpy(force=True))

                y_true = np.concatenate(y_true, axis=0, dtype=np.int32, casting="unsafe") # numpy array of true labels
                y_pred = np.concatenate(y_pred, axis=0, dtype=np.int32, casting="unsafe") # numpy array of predicted labels
                print("\n", dataloader.lang.upper(), "DATASET")
                print(classification_report(y_true, y_pred, target_names=labels, zero_division=np.nan)) # scikit-learn classification report
                # for label, conf_mat in zip(labels, multilabel_confusion_matrix(y_true, y_pred)):
                #     print(f"{label}:\n{conf_mat}\n")
        
    
    def fit(self, train_loaders, valid_loaders, epochs, lr=1e-5, loss_target=0.0, save_path="./ckpts/", resume_from=None): # Adam optimizer, BCE loss
        return self.model.fit(Adam, {"lr": lr}, BCEWithLogitsLoss, {}, train_loaders, valid_loaders, epochs, loss_target, save_path, resume_from)
    
    def classify(self, text): # classifies text into boolean values for each label
        if type(text) != str: raise TypeError("text must be a string")
        self.eval()
        with torch.inference_mode():
            pred_labels =  self.model(text, batch_mode=False)
            return (pred_labels > 0).any(dim=0).tolist()
        
    def __call__(self, text):
        return self.classify(text)
    
    def predict_proba(self, sequences): # takes a list of strings and returns a numpy array of probabilities for each label
        self.eval()
        with torch.inference_mode():
            pred_labels = self.model(sequences)
            return pred_labels.sigmoid().numpy(force=True) # shape=(batch_size, num_labels)
    
    def save(self, path, comment=None):
        self.model.save(path, comment)

    def load(self, path):
        return self.model.load(path)
    
    @classmethod
    def from_trained(cls, path, print_comment=False, **kwargs):
        return cls(path=path, print_comment=print_comment, **kwargs)