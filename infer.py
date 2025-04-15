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


# test_data = load_from_dir("./test", BATCH_SIZE)

MODEL_PATH = R"D:\Misc\xlm-roberta.pth"

if __name__ == "__main__":
    print("\nLoading model from", MODEL_PATH)
    model = EmotionClassifier.from_trained(MODEL_PATH, True)
    model.to("cuda")

    # model.evaluate(test_data)

    CLSCR()
    while(text := input(U_LINE_TEXT("Enter text") + ": ")):

        pred_labels = model(text)
        emotions = ",".join(LABELS[pred_labels])
        print(f"Detected emotions: {HIGHLIGHT_TEXT(emotions)}\n")

    RESET_TERM()
    CLSCR()