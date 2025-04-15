from os import walk
from os.path import join
from EmotionModels import EmotionClassifier, EmotionDataset, EmotionDataloader

TRAIN_DIR = "./train"
VALID_DIR = "./valid"

# note: max batch size for 4gb vram is 2
# note: max batch size for 8gb vram is 8
# note: max batch size for tesla T4(16gb vram) is 64
BATCH_SIZE = 2

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
    # model = EmotionClassifier("FacebookAI/xlm-roberta-base") # from_pretrained=True)
    # model.load(R"D:\Misc\model_32_181509.ckpt")
    model = EmotionClassifier.from_trained(R"D:\Misc\xlm-roberta.pth", True)
    model.to("cuda")

    # model.fit(train_data, valid_data, epochs=35, lr=1e-5, resume_from="/content/ckpts/model_20_145320.ckpt")

    # model.evaluate(train_data)
    model.evaluate(valid_data)

#  DEU DATASET
#               precision    recall  f1-score   support

#        anger       0.77      0.52      0.62       151
#      disgust       0.62      0.61      0.61       161
#         fear       0.59      0.20      0.30        50
#          joy       0.66      0.61      0.64       114
#      sadness       0.56      0.50      0.53       103
#     surprise       0.29      0.27      0.28        37

#    micro avg       0.62      0.52      0.56       616
#    macro avg       0.58      0.45      0.50       616
# weighted avg       0.63      0.52      0.56       616
#  samples avg       0.39      0.37      0.37       616


#  ENG DATASET
#               precision    recall  f1-score   support

#        anger       0.31      0.43      0.36        60
#         fear       0.77      0.74      0.75       336
#          joy       0.73      0.43      0.54       146
#      sadness       0.58      0.70      0.64       189
#     surprise       0.58      0.60      0.59       161

#    micro avg       0.64      0.63      0.64       892
#    macro avg       0.60      0.58      0.58       892
# weighted avg       0.66      0.63      0.64       892
#  samples avg       0.55      0.57      0.53       892


#  ESP DATASET
#               precision    recall  f1-score   support

#        anger       0.76      0.62      0.68       100
#      disgust       0.78      0.78      0.78       120
#         fear       0.60      0.81      0.69        47
#          joy       0.84      0.88      0.86       130
#      sadness       0.63      0.95      0.75        60
#     surprise       0.84      0.64      0.73        75

#    micro avg       0.75      0.78      0.76       532
#    macro avg       0.74      0.78      0.75       532
# weighted avg       0.77      0.78      0.76       532
#  samples avg       0.79      0.81      0.77       532