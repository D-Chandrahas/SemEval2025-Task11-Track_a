import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from BaseModels import Roberta, ClassificationHead, EncoderClassifier
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from collections import defaultdict


class EmotionDataset(Dataset):
    cols = np.array(("text", "anger", "disgust", "fear", "joy", "sadness", "surprise"))
    labels = cols[1:]
    def __init__(self, path):
        (dtypes := defaultdict(lambda: np.float32))["text"] = str
        self.data = pd.read_csv(path, header=0, dtype=dtypes)
        self.data = self.data.reindex(columns=self.cols, fill_value=np.float32(-1))
        self.loss_mask = [(x >= 0) for x in self.data.iloc[0, 1:]]
        self.labels = self.__class__.labels[self.loss_mask]
        self.lang = path[-7:-4]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index, 0], torch.tensor(self.data.iloc[index, 1:].array, dtype=torch.float32)
    

class EmotionDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1):
        self.loss_mask = dataset.loss_mask
        self.labels = dataset.labels
        self.lang = dataset.lang
        super().__init__(dataset, batch_size)


class EmotionClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        encoder = Roberta("FacebookAI/xlm-roberta-base")
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
        with torch.no_grad():
            for dataloader in data:
                y_true, y_pred = [], []
                loss_mask = dataloader.loss_mask
                labels = dataloader.labels

                for sequences, true_labels in dataloader:
                    y_true.append((true_labels[:, loss_mask]).numpy(force=True))
                    pred_labels = self.model(sequences)
                    y_pred.append((pred_labels[:, loss_mask] > 0).numpy(force=True))

                y_true = np.concatenate(y_true, axis=0, dtype=np.int32, casting="unsafe")
                y_pred = np.concatenate(y_pred, axis=0, dtype=np.int32, casting="unsafe")
                print("\n", dataloader.lang, "dataset")
                print(classification_report(y_true, y_pred, target_names=labels, zero_division=0.0))
                # for label, conf_mat in zip(labels, multilabel_confusion_matrix(y_true, y_pred)):
                #     print(f"{label}:\n{conf_mat}\n")
        
    
    def fit(self, train_data, valid_data, epochs, lr=1e-5, loss_target=0.0, save_path="./ckpts/", resume_from=None):
        return self.model.fit(Adam, {"lr": lr}, BCEWithLogitsLoss, {}, train_data, valid_data, epochs, loss_target, save_path, resume_from)
    
    def classify(self, text):
        if type(text) != str: raise TypeError("text must be a string")
        self.eval()
        with torch.no_grad():
            pred_labels =  self.model(text, batch_mode=False)
            return (pred_labels > 0).any(dim = 0).tolist()
        
    def __call__(self, text):
        return self.classify(text)
    
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        return self.model.load(path)