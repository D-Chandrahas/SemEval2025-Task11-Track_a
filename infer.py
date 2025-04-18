print("\nLoading Modules...")
from EmotionModels import EmotionClassifier, EmotionDataset
from train import load_from_dir, BATCH_SIZE
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

LABELS = EmotionDataset.labels

CLSCR = lambda : print("\x1b[2J\x1b[3J\x1b[1;1H", end="")
RESET = "\x1b[0m"
RESET_TERM = lambda : print(RESET, end="")
U_LINE = "\x1b[4m"
HIGHLIGHT = "\x1b[7m"
RED = "\x1b[30;101m"
GREEN = "\x1b[30;102m"
U_LINE_TEXT = lambda s : U_LINE + s + RESET
HIGHLIGHT_TEXT = lambda s : HIGHLIGHT + s + RESET
RED_TEXT = lambda s : RED + s + RESET
GREEN_TEXT = lambda s : GREEN + s + RESET


test_data = load_from_dir("./test", BATCH_SIZE)

MODEL_PATH = R"D:\Misc\xlm-roberta1.pth"

if __name__ == "__main__":
    print("\nLoading model from", MODEL_PATH)
    model = EmotionClassifier.from_trained(MODEL_PATH, True)
    model.to("cuda")

    model.evaluate(test_data)

    # CLSCR()
    # while(text := input(U_LINE_TEXT("Enter text") + ": ")):

    #     pred_labels = model(text)
    #     emotions = ",".join(LABELS[pred_labels])
    #     print(f"Detected emotions: {HIGHLIGHT_TEXT(emotions)}\n")

    # RESET_TERM()
    # CLSCR()

#  DEU DATASET
#               precision    recall  f1-score   support

#        anger       0.72      0.80      0.76       161
#      disgust       0.63      0.85      0.72       168
#         fear       0.31      0.35      0.33        43
#          joy       0.71      0.75      0.73       116
#      sadness       0.65      0.42      0.51       100
#     surprise       0.15      0.58      0.24        33

#    micro avg       0.57      0.70      0.63       621
#    macro avg       0.53      0.62      0.55       621
# weighted avg       0.62      0.70      0.65       621
#  samples avg       0.48      0.52      0.48       621


#  ENG DATASET
#               precision    recall  f1-score   support

#        anger       0.55      0.27      0.36        64
#         fear       0.79      0.52      0.63       327
#          joy       0.68      0.59      0.63       138
#      sadness       0.75      0.56      0.64       192
#     surprise       0.74      0.62      0.68       180

#    micro avg       0.74      0.54      0.63       901
#    macro avg       0.70      0.51      0.59       901
# weighted avg       0.74      0.54      0.62       901
#  samples avg       0.65      0.51      0.55       901


#  ESP DATASET
#               precision    recall  f1-score   support

#        anger       0.54      0.92      0.68        90
#      disgust       0.71      0.92      0.80       132
#         fear       0.88      0.82      0.85        45
#          joy       0.92      0.76      0.83       120
#      sadness       0.81      0.81      0.81        57
#     surprise       0.78      0.66      0.72        77

#    micro avg       0.73      0.82      0.77       521
#    macro avg       0.77      0.81      0.78       521
# weighted avg       0.77      0.82      0.78       521
#  samples avg       0.78      0.84      0.79       521