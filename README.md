# Ukrainian emotions classification (NLP Course competition)

**Author**: Mark Matviiv, PKN24/M\
2025, UCU

## Overview
This is a report of the [Ukrainian emotions classification task](https://www.kaggle.com/competitions/ucu-ukrainian-emotions/overview).

The goal of the task is to classify short Ukrainian reviews by categories and emotions. Solution should be based on NN and not use APIs. Evaluation metric is Macro F1 averaged across tasks (columns).

Files:
- [`train.csv`](./train.csv): dataset for training
- [`test.csv`](./test.csv): competition dataset
- [`submission.csv`](./submission.csv): my submission file for the competition
- [`run.ipynb`](./run.ipynb): notebook with the solution
- [`models/`](./models/): directory for model training (empty cuz each fold is more than 1Gb)

## Approach
The main idea is to use a Bert-based model which is pre-trained on a large dataset of Ukrainian text. Then we train a two-headed model to classify the reviews by categories and emotions.

I tried different models (xlm-roberta-base, youscan/ukr-roberta-base, etc.) expecting that the `ukr-roberta-base` will be the best model since it ukranian-specific model. However, the best results were achieved with the `microsoft/mdeberta-v3-base` which is multilingual model.

To achive the best results, I've use the K-Fold cross-validation with 5 folds.
For each fold, I've trained a model and saved the best model.
Then I've used the best model to predict the test set.

## Challenges
The main problem is the class imbalance in the dataset.
![Class distribution](./img/distribution.png)

As you can see the amount of samples for each class is far from equal.
Especially for the `Emotion` labels. That cause a lot of problems for the model to learn the `disgust`, `suprise`, `fear` classes.
![Confusion matrix](./img/matrix.png)

I tried a lot of optimization techniques and tricks to improve the results for them. But without generating some additional data for the rare classes, it's kinda impossible.

Since we cannot use any API to use LLMs for generating data, I decided to accept the results as is.

The only tricks I decided to stick with is the Focal Loss and weights smoothing for slightly compensating the imbalance.

## Results

The avarege F1 scores I got was:

| Metric | Emotion | Category | Overall |
|--------|---------|----------|---------|
| F1     | 0.56    | 0.80     | 0.68    |

Which is pretty good taking into account the class imbalance. Hight category score compensates a bit the low emotion one.
