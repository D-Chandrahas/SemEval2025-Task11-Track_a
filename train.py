from os import walk
from os.path import join
from EmotionModels import EmotionClassifier, EmotionDataset, EmotionDataloader

TRAIN_DIR = "./train"
VALID_DIR = "./valid"

# note: max batch size for rtx 3050 mobile(4gb vram) is 4
# note: max batch size for tesla T4(16gb vram) is 16
BATCH_SIZE = 4

def load_from_dir(dir, batch_size=1):
    data = []
    try:
        for path in (join(dir, file) for file in next(walk(dir))[2]):
            data.append(
                EmotionDataloader(
                    EmotionDataset(path),
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
    model = EmotionClassifier()
    # model.load("D:/Misc/model_2_130059.ckpt")
    model.to("cuda")

    model.fit(train_data, valid_data, epochs=10, save_path="R:/")

    # print(model.evaluate(train_data))
    # print(model.evaluate(valid_data))
