# UCAS_DL2023_Spring
This is a repository for "Deep Learning 2023 spring" course in UCAS.

In this repository, there are four projects that handwrited number recognition, car & dog classification, automatic poetry, movie emotion classification. Each of them will be implemented in a unified framework. Until now, handwrited number recognition, car & dog classification are implemented already...

## Implementation
handwrited number recognition   √
car & dog classification        √
automatic poetry                ×
movie emotion classification    ×

## Usage
run.py is the api for model training, testing. In `MyDataset.py`, you can change the data path into your own one. To get the mnist data, you can look at the `datascript.py', and use the function in there.

```
python run.py --experiment mnist --mode train
```

### Input parameters:
* `--experiment`: specifies one of the four experiment (mnist, kaggle).
* `--mode`: specifies the mode of pipeline (train, test, pred(for kaggle)).
* `--path`: specifies the path of model parameters.
* `--early_stopping`: training procedure with early stopping or not.
* `--early_stopping`: specifies the patient of early stopping.
* `--cuda`: using GPU or not.
* `--lr`: specifies the learning rate.
* `--bs`: specifies the batch size.
* `--es`: sepcified the epoch size.
* `--tr`: sepcified the rate of traning vs val data split.
* `--net`: for kaggle, specifies the pretrained net architecture you want to use (resnet18, resnet50). If do not specifies, it will use simple CNN architecture defined in `MyModel.py`.
* `--output`: for kaggle, you can use it to specify the pred result path.

