import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import os
from datetime import datetime
import wandb
import random
import numpy as np
import ssl

from data.dataset_utils import load_cifar10
from models.model_utils import get_models

from config import get_config

ssl._create_default_https_context = ssl._create_unverified_context

random_state = get_config("general", "random_state")
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)

def train():
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(f"outputs/{now}")
    
    torch.multiprocessing.freeze_support()
    config_gen = get_config("general")

    if config_gen["device"] == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config_gen["device"])

    trainloader, testloader = load_cifar10()
    num_epochs = config_gen["num_epochs"]

    for model, config in get_models():
        os.makedirs(f"outputs/{now}/"+config["name"])
        wandb.init(project="SampleImageClassification_"+config["name"], config=get_config("wandb"))
        model = model.to(device)
        
        train_settings = config["train_settings"]
        loss_function = train_settings["loss_function"]
        optimizer = train_settings["optimizer"](model.parameters())

        for epoch in range(num_epochs):
            results = ""
            running_loss = 0.0
            print(f"Epoch: {epoch+1}")
            for i, data in tqdm(enumerate(trainloader, 0)):
                x, y = data
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = loss_function(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # 各エポック後の処理
            with torch.no_grad():
                running_score = 0.0
                for j, data in tqdm(enumerate(testloader, 0)):
                    x, y = data
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    running_score += config["train_settings"]["eval_function"](pred, y)
            epoch_loss, epoch_score = running_loss/(i+1), running_score/(j+1)    
            wandb.log({"Loss":epoch_loss, "Score":epoch_score})   
            result = f"Loss: {epoch_loss}  Score: {epoch_score}\n"
            results += ("Epoch:"+str(epoch+1)+"  "+result)
            print(result)
            
        # モデル学習完了後の処理
        out_dir = f"outputs/{now}/"+config["name"]+"/"
        torch.save(model.state_dict(), out_dir+"model.pth")
        with open(out_dir+"results.txt", "w") as file:
            file.write(results)
        wandb.finish()
        print("Training finished")
        

if __name__ == "__main__":
    train()