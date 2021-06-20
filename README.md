# StableReprLearning

This repository is the official implementation of [Stable Representation Learning for Neural Network](https://arxiv.org/xxx).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Run Experiments

To `train` and `evaluate` the model(s) with `StableReprLearning` framework in the paper, run this command:

```train
python -m entry.srl_framework.<model> <number_of_experiments: 15> <early_stop_patience: 10>
```

To `train` and `evaluate` the model(s) with `Normal` framework, run this command:

```train
python -m entry.normal_framework.<model> <number_of_experiments: 15> <early_stop_patience: 10>
```

<!-- ## Results

Our model achieves the following performance on :

## Contributing -->
