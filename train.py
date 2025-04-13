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
    model = EmotionClassifier()
    model.load(R"D:\Misc\model_20_145320.ckpt")
    model.to("cuda")

    # model.fit(train_data, valid_data, epochs=10)

    # model.evaluate(train_data)
    model.evaluate(valid_data)
