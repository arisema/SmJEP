import torch
from omegaconf import OmegaConf


import k_diffusion as K
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import optuna

from utils import get_model, instantiate_from_config


def train(trial):
    config_file = "/ocean/projects/cis230036p/amihretu/SmJEP/architecture/model/configs/train.yaml"
    
    train_config = OmegaConf.load(config_file)
    
    ## TEST DIFFERENT HPARAM VALUES
    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True) 
    # batch_size = trial.suggest_categorical("batch_size", [16, 32])

    ### init loggers and checkpoint callback
    run_name = datetime.now().strftime("%Y_%m_%d__%H_%M")+"SN1"+"_{epoch:02d}"#+f"__lr_{learning_rate}__bs_{batch_size}"
    ## log to WANDB
    wandb_logger = WandbLogger(project='SmJEP', log_model=all, name=run_name)
    ## save checkpoints
    checkpoint_callback = instantiate_from_config(train_config.lightning.callbacks.checkpoint_callback)
    checkpoint_callback.filename = 'SubNet1-'+run_name

    gpus = torch.cuda.device_count()

    trainer = Trainer.from_argparse_args(train_config.lightning.subnet1_trainer, gpus=gpus, logger=wandb_logger, callbacks=[checkpoint_callback])
    
    ## Turn this into load from config and read from lightning
    data = instantiate_from_config(train_config.data)
    data.dataloader_type = 'subnet1'
    model = get_model(train_config.SubNet1)
    
    # lr_finder = trainer.tuner.lr_find(model, datamodule=data, min_lr=1e-05, max_lr=1e-03)
    # import pprint
    # with open('bs_results', 'w') as f:
    #     f.write(f"{vars(lr_finder)}")
    # model.learning_rate = learning_rate
    # data.batch_size = batch_size
  
    trainer.fit(model, data)

    # trainer.test(model, data)
    return trainer.callback_metrics['val/loss'].item()

 
if __name__ == '__main__':

    # train()

    seed_everything(42)

    # Create an Optuna study object and optimize
    study = optuna.create_study(direction="minimize")  # Adjust based on your optimization goal
    study.optimize(train, n_trials=10)  # Adjust the number of trials as needed

    # Print the best hyperparameters and result
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
