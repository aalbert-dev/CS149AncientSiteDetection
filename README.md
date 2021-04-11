# Project 1 - Ancient Site Detection

Binary image classification model for ancient site detection.

## Train

```bash
cd model_simple_project && \
python train.py --root-dir /content/ --num-epochs 100
```

## Predict

```bash
cd model_simple_project && \
python eval.py --root-dir /content --model model.pt
```

## Predictions for given test set

See `model_simple_project/predcit_result.csv`.
