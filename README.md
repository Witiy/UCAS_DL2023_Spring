# UCAS_DL2023_Spring
This is a repository for "Deep Learning 2023 spring" course in UCAS.

In this repository, there are four projects that handwrited number recognition, car & dog classification, automatic poetry, movie emotion classification. Each of them will be implemented in a unified framework. 

## Implementation
handwrited number recognition   √

car & dog classification        √

automatic poetry                √

movie emotion classification    √

## Usage
run.py is the api for model training, testing. In `MyDataset.py`, you can change the data path into your own one. To get the mnist data, you can look at the `datascript.py`, and use the function in there.

```
python run.py --experiment mnist --mode train
python run.py --experiment mnist --mode test

python run.py --experiment kaggle --mode train --net resnet18
python run.py --experiment kaggle --mode test --net resnet18

python run.py --experiment tange --mode train --note gan
python run.py --experiment mnist --mode gen --begin 床前明月光

python run.py --experiment mnist --mode train
python run.py --experiment mnist --mode test
```

### Input parameters:
* `--experiment`: specifies one of the four experiment (mnist, kaggle, tang, movie).
* `--mode`: specifies the mode of pipeline (train, test, gen(for tang)).
* `--path`: specifies the path of model parameters.
* `--early_stopping`: training procedure with early stopping or not.
* `--patient`: specifies the patient of early stopping.
* `--cuda`: using GPU or not.
* `--lr`: specifies the learning rate.
* `--bs`: specifies the batch size.
* `--es`: sepcified the epoch size.
* `--tr`: sepcified the rate of traning vs val data split.
* `--net`: for kaggle, specifies the pretrained net architecture you want to use (resnet18, resnet50). If do not specifies, it will use simple CNN architecture defined in `MyModel.py`.
* `--note`: for tang, I use GAN as another loss function to train the network beside autoregressive loss function. (default: gan)
* `--begin`: for tang, input some incomplete poem, and the trained model will generate the whole poem. NOTE: should use under mode:gen.

Also, you can use the `example.py` in `notebook` dir as tutorial.
