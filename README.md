# UCAS_DL2023_Spring
This is a repository for "Deep Learning 2023 spring" course in UCAS.

In this repository, there are four projects that handwrited number recognition, car & dog classification, automatic poetry, movie emotion classification. Each of them will be implemented in a unified framework. Until now, handwrited number recognition, car & dog classification are implemented already...

## Implementation
handwrited number recognition   √
car & dog classification        √
automatic poetry                ×
movie emotion classification    ×

## Usage
run.py is the api for model training, testing. In MyDataset.py, you can change the data path into your own one.

```
python run.py --experiment mnist
```

## TODO
fix tqdm as https://github.com/BohriumKwong/pytorch_use_demo/blob/master/README.md
