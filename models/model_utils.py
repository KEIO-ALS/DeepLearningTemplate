from models.SimpleCNN import SimpleCNN
from models.ResNet import ResNet18

import sys
sys.path.append('../')
from config import get_config

models = {
    "SimpleCNN": SimpleCNN,
    "ResNet": ResNet18,
}     
    
# stateがTrueのモデルとその設定一覧を返す
def get_models():
    selected_models = []
    for key in models:
        model_config = get_config("models")[key]
        if model_config["state"]:
            selected_models.append([models[key](model_config["param"]), model_config])
    return selected_models