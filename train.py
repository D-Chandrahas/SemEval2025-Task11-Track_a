from os import walk
from os.path import join
from EmotionModels import EmotionClassifier, EmotionDataset, EmotionDataloader
from torch.cuda import is_available as cuda_available
DEVICE = "cuda" if cuda_available() else "cpu"

TRAIN_DIR = "./train"
VALID_DIR = "./valid"

# note: max batch size for 4gb vram is 2
# note: max batch size for 8gb vram is 8
# note: max batch size for tesla T4(16gb vram) is 64
BATCH_SIZE = 2

# loads all files in a directory and returns a list of dataloaders
def load_from_dir(dir, batch_size=1):
    data = []
    try:
        for file in next(walk(dir))[2]:
            data.append(
                EmotionDataloader(
                    EmotionDataset(join(dir, file)),
                    batch_size
                )
            )
    except StopIteration:
        raise NotADirectoryError(f"{dir} is not a valid directory")
    
    if len(data) == 0:
        raise FileNotFoundError(f"No files found in {dir}")
    
    return data



train_data = load_from_dir(TRAIN_DIR, BATCH_SIZE)
valid_data = load_from_dir(VALID_DIR, BATCH_SIZE)


if __name__ == "__main__":
    # model = EmotionClassifier("FacebookAI/xlm-roberta-base", from_pretrained=True) # create pretrained model from huggingface

    # model = EmotionClassifier("FacebookAI/xlm-roberta-base") # create new model
    # model.load(R"D:\Misc\model_5_170139.ckpt") # load from training checkpoint file

    model = EmotionClassifier.from_trained(R"D:\Misc\xlm-roberta1.pth", True) # load from trained model file


    model.to(DEVICE)


    # model.fit(train_data, valid_data, epochs=5, lr=1e-5)
    # model.save(R"D:\Misc\xlm-roberta1.pth", "Epochs=5; BCEWithLogitsLoss: train_loss<0.3, valid_loss=0.39; optimizer: Adam, lr=1e-5; batch_size=4; train_data(0.8, sorted)=[deu, eng, esp]; random_state=7;")


    # model.evaluate(train_data)
    model.evaluate(valid_data)


#  DEU DATASET
#               precision    recall  f1-score   support

#        anger       0.66      0.78      0.72       151
#      disgust       0.57      0.79      0.66       161
#         fear       0.39      0.44      0.42        50
#          joy       0.66      0.68      0.67       114
#      sadness       0.64      0.50      0.57       103
#     surprise       0.25      0.76      0.37        37

#    micro avg       0.55      0.69      0.61       616
#    macro avg       0.53      0.66      0.57       616
# weighted avg       0.59      0.69      0.62       616
#  samples avg       0.48      0.51      0.48       616


#  ENG DATASET
#               precision    recall  f1-score   support

#        anger       0.64      0.45      0.53        60
#         fear       0.89      0.57      0.69       336
#          joy       0.81      0.55      0.66       146
#      sadness       0.75      0.51      0.61       189
#     surprise       0.65      0.63      0.64       161

#    micro avg       0.78      0.56      0.65       892
#    macro avg       0.75      0.54      0.63       892
# weighted avg       0.79      0.56      0.65       892
#  samples avg       0.65      0.53      0.56       892


#  ESP DATASET
#               precision    recall  f1-score   support

#        anger       0.68      0.86      0.76       100
#      disgust       0.69      0.87      0.77       120
#         fear       0.93      0.89      0.91        47
#          joy       0.90      0.76      0.82       130
#      sadness       0.87      0.77      0.81        60
#     surprise       0.76      0.79      0.77        75

#    micro avg       0.77      0.82      0.80       532
#    macro avg       0.80      0.82      0.81       532
# weighted avg       0.79      0.82      0.80       532
#  samples avg       0.82      0.85      0.81       532