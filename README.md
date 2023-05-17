# DeepLearningTemplate
A basic deep learning model training environment running image classification models on the CIFAR-10 dataset, build using PyTorch (2.0.1).

## Recommended Development Environment

## Environment building
### Step1. Build a virtual environment
Build a virtual environment in an arbitrary working directory using python's venv.

`cd /path/to/your/working/directory`

`python -m venv venv`

`venv\Scripts\activate.dat`

### Step2. Clone Repository
Clone the repository to your local directory.

`cd venv`

`git clone https://github.com/KEIO-ALS/DeepLearningTemplate.git`

`cd DeepLearningTemplate`

Add a new remote repository. In the following command, 
the new remote repository is named "new-origin",
but can be named arbitarily.
Replace "https://github.com/yourusername/new-repo.git" with the URL of your new GitHub repository.

`git remote add new-origin https://github.com/yourusername/new-repo.git`

`git add .`

`git commit -m "first"`

`git push new-origin master --force`

`git remote remove origin`


### Step3. Library Installation
Install the libraries required to run.

`..\Scripts\python.exe -m pip install -r requirements.txt`

Use Weights & Biases to observe loss and accuracy trends online
1. Visit https://wandb.ai/site and sign up
2. `wandb login`


## Training Sample Model

`python run.py`


## Using Docker
`pip freeze > requirements.txt`

`docker build -t image_name .`

`docker run -it -v ./outputs:/app/outputs image_name`





