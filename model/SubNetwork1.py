import torch
from omegaconf import OmegaConf
from utils import instantiate_from_config

import k_diffusion as K
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


def get_model(config):
    
    checkpoint = torch.load(config.model.params.ckpt_path)
    model = instantiate_from_config(config.model)
    _,_ = model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.learning_rate = config.model.base_learning_rate
    return model

def train():
    config_file = "/ocean/projects/cis230036p/amihretu/SmJEP/architecture/model/configs/train.yaml"
    
    train_config = OmegaConf.load(config_file)

    print("="*10)

    data = instantiate_from_config(train_config.data)
    model = get_model(train_config.SubNet1)
    ## log to WANDB
    # wandb_logger = WandbLogger(project='SmJEP', log_model=all)

    gpus = torch.cuda.device_count()
    trainer = Trainer.from_argparse_args(train_config.lightning.trainer, gpus=gpus, )#, logger=wandb_logger)

    ## Turn this into load from config and read from lightning
    data = instantiate_from_config(train_config.data)
    model = get_model(train_config.SubNet1)
    
    trainer.fit(model, data)

    trainer.test(model, data)


 
if __name__ == '__main__':
    
    train()
    
