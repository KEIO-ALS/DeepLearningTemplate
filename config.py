import torch
import torch.nn as nn
import torch.optim as optim

from models.validation_functions import get_classification_accuracy

c = {
    "general":{
        "num_epochs": 10,
        "random_state": 111,
        "batch_size": 3,
        "num_workers": 2,
        "device": "cuda",
        "test_size": 0.2,
    },
    "data":{
  
    },
    "models":{
        "SimpleCNN":{
            "name": "SimpleCNN",
            "state": False,
            "train_settings":{
                "loss_function": nn.CrossEntropyLoss(),
                "optimizer": optim.Adam,
                "eval_function": get_classification_accuracy,
            },
            "param":{},        
        },
        "Seq2Seq":{
            "name": "Seq2Seq",
            "state": True,
            "train_settings":{
                "loss_function": nn.CrossEntropyLoss(),
                "optimizer": optim.Adam,
                "eval_function": get_classification_accuracy,
            },
            "param":{
                "input_dim": 13,
                "output_dim": 13,
                "hidden_dim": 32,
                "num_layers": 2,
            },        
        },
    },
    "wandb":{
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
}

def get_config(*keys):
    config = c
    for key in keys:
        config = config[key]
    return config 