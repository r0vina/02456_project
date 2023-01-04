# 02456_project - Structure Tensor Hyper-resolution

This GitHub repo contains all code used for our project in the DTU course 02456 - Deep Learning.

The repository structure is as follows:

**exploration_ipynbs** contains jupyter notebooks that were used to develop code related to data processing and model configuration/training, which could then be used to construct .py and .sh scripts to use on DTU's HPC.

**runners** contains the .sh scripts we ran on HPC. The scripts differ only slightly in terms of what trainer script they use.

**trainers** contains the .py scripts that contain our whole pipeline from data processing/transformation to model creation, training, validation, testing, and saving of the model.
- We will in the future revisit this script to divide and refactorise into separate scripts in order to allow for easier reproducibility and maintainability, etc.

**utilities** contains minor utility scripts and a requirements.txt file that contains the required packages to run a trainer script.

**wandb** contains folders and files only relevant for wandb, and these are therefore sort of a blackbox.

### Repo tree:
```bash
├── exploration_ipynbs
│   ├── DLProject.ipynb
│   ├── LoadModel.ipynb
├── runners
│   ├── submit_train_test.sh
│   ├── submit_train_test_run.sh
├── trainers
│   ├── train_script.py
│   ├── train_script_run.py
├── utilities
│   ├── image_cutter.py
│   ├── model2.py
│   ├── rename-files.py
│   ├── requirements.txt
├── wandb
│   ├── ...
├── .gitignore
└── README.md
```

Data is accesible on Google drive: https://drive.google.com/drive/folders/1Nl0ThLbXHwq59G8rAe5zgxeDpUX8T3nu?usp=share_link
