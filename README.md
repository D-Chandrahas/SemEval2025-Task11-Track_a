
**note: make sure the working directory is set to the root folder of the codebase.**

The trained model file can be downloaded [here](https://iith-my.sharepoint.com/:u:/g/personal/ai21btech11010_iith_ac_in/ERlwrsMLft1IpF4P02TJrzkBLyW0TV_T691iQBSAKpjxOw?e=R2btyv).

## Preprocessing
- run `python preprocess.py` to generate test, train & valid datasets.

## Training and Validation
- in `train.py` file:
	- set appropriate `BATCH_SIZE`.
	- in `__main__` block:
		- Uncomment appropriate lines depending on how you want to create/load the model. Modify the path parameter accordingly.
		- Uncomment `model.fit` & `model.save` and modify parameters based on requirement.
		- Uncomment `model.evaluate` based on requirement.
- run `python train.py`.

## Inference
- in `infer.py`:
	- modify `MODEL_PATH` to the location of downloaded model file.
	- (if required) uncomment `model.evaluate` and comment out cli code to evaluate on test dataset.
	- (if required) modify `DEVICE` to force cpu backend.
- run `python infer.py` to get cli.

## Visualization
- modify and run `visualization.ipynb` notebook accordingly as per requirement.

## Evaluation of test dataset

### DEU Dataset

| Emotion   | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Anger     | 0.72      | 0.80   | 0.76     | 161     |
| Disgust   | 0.63      | 0.85   | 0.72     | 168     |
| Fear      | 0.31      | 0.35   | 0.33     | 43      |
| Joy       | 0.71      | 0.75   | 0.73     | 116     |
| Sadness   | 0.65      | 0.42   | 0.51     | 100     |
| Surprise  | 0.15      | 0.58   | 0.24     | 33      |
|||||
| **Micro Avg** | **0.57** | **0.70** | **0.63** | **621** |
| **Macro Avg** | **0.53** | **0.62** | **0.55** | **621** |
| **Weighted Avg** | **0.62** | **0.70** | **0.65** | **621** |
| **Samples Avg** | **0.48** | **0.52** | **0.48** | **621** |

---
### ENG Dataset

| Emotion   | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Anger     | 0.55      | 0.27   | 0.36     | 64      |
| Fear      | 0.79      | 0.52   | 0.63     | 327     |
| Joy       | 0.68      | 0.59   | 0.63     | 138     |
| Sadness   | 0.75      | 0.56   | 0.64     | 192     |
| Surprise  | 0.74      | 0.62   | 0.68     | 180     |
|||||
| **Micro Avg** | **0.74** | **0.54** | **0.63** | **901** |
| **Macro Avg** | **0.70** | **0.51** | **0.59** | **901** |
| **Weighted Avg** | **0.74** | **0.54** | **0.62** | **901** |
| **Samples Avg** | **0.65** | **0.51** | **0.55** | **901** |

---
### ESP Dataset

| Emotion   | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Anger     | 0.54      | 0.92   | 0.68     | 90      |
| Disgust   | 0.71      | 0.92   | 0.80     | 132     |
| Fear      | 0.88      | 0.82   | 0.85     | 45      |
| Joy       | 0.92      | 0.76   | 0.83     | 120     |
| Sadness   | 0.81      | 0.81   | 0.81     | 57      |
| Surprise  | 0.78      | 0.66   | 0.72     | 77      |
|||||
| **Micro Avg** | **0.73** | **0.82** | **0.77** | **521** |
| **Macro Avg** | **0.77** | **0.81** | **0.78** | **521** |
| **Weighted Avg** | **0.77** | **0.82** | **0.78** | **521** |
| **Samples Avg** | **0.78** | **0.84** | **0.79** | **521** |