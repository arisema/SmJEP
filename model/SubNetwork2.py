import torch
from omegaconf import OmegaConf
from utils import instantiate_from_config, get_model, decode_motor_commands, task_completion_metric

import k_diffusion as K
from pytorch_lightning import Trainer, seed_everything, LightningModule
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import optuna

import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnet50


class SubNetwork2(LightningModule):
    def __init__(self, 
                 image_extractor_out_dim: int=256,
                 hidden_size=32,
                 lstm_num_layers=2,
                 embedding_size=8,
                 motor_command_max_length=11,
                 vocab_size=9,
                 learning_rate=2e-03):
        super().__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.motor_command_max_length = motor_command_max_length
        self.lstm_num_layers = lstm_num_layers

        # CNN feature extractor
        self.image_feature_extractor = resnet50(pretrained=True)
        self.image_feature_extractor.fc = nn.Linear(in_features=2048, out_features=image_extractor_out_dim)
        
        self.sequence_processing = nn.LSTM(image_extractor_out_dim, hidden_size, lstm_num_layers, batch_first=True)

        # Decoder
        ## Embedding 
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.decoder = nn.LSTMCell(embedding_size, hidden_size)
        self.decoder_head = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.loss_function = nn.CrossEntropyLoss()

        self.init_weights()

    def init_weights(self):
        for name, param in self.sequence_processing.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)

        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)

        
    def forward(self, start_scene_image, goal_scene_image, motor_cmd_length=None):
        start_scene_feats = self.image_feature_extractor(start_scene_image)
        goal_scene_feats = self.image_feature_extractor(goal_scene_image)

        combined_images = torch.stack([start_scene_feats, goal_scene_feats], dim=1)
        # Encoder forward pass
        _, (h, c) = self.sequence_processing(combined_images)
        ## Get last layer hidden value
        # print(f"===\nh: {h.shape}\n{h}\n===")

        h, c = h[-1,:,:], c[-1,:,:]

        # print(f"===\nh: {h.shape}\n{h}\n===")

        motor_command_max_length = self.motor_cmd_length if motor_cmd_length is None else motor_cmd_length
        batch_size = start_scene_feats.shape[0]

        motor_command_output = []
        motor_command_tensor = torch.ones(batch_size).long().to(self.device)

        # Decoder forward pass
        for t in range(motor_command_max_length):

            motor_command_tensor = self.embedding(motor_command_tensor).squeeze()

            h, c = self.decoder(motor_command_tensor, (h,c))
            
            output = self.decoder_head(h)
            
            output = self.softmax(output)

            motor_command_output.append(output.unsqueeze(1))

            motor_command_tensor = output.argmax(dim=1).unsqueeze(0)

        motor_command_output = torch.cat(motor_command_output, 1)
        
        return motor_command_output

    def get_inference_inputs(self, batch):
        # training_step defines the train loop.
        start_scene_images = batch['start_state_image']
        goal_scene_images = batch['goal_state_image']
        target_caption = batch['motor_command']

        motor_command_max_length = target_caption.shape[1]

        return start_scene_images, goal_scene_images, target_caption, motor_command_max_length

    def training_step(self, batch, batch_idx):
        start_scene_images, goal_scene_images, target_caption, motor_command_max_length = self.get_inference_inputs(batch)

        motor_command_output = self(start_scene_images, goal_scene_images, motor_command_max_length) # call forward function

        # Compute loss if target_caption is provided
        
        motor_command_output = motor_command_output.view(-1, motor_command_output.size(2))

        loss = self.loss_function(motor_command_output, target_caption.view(-1))
        self.log("train/loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        start_scene_images, goal_scene_images, target_caption, motor_command_max_length = self.get_inference_inputs(batch)

        motor_command_output = self(start_scene_images, goal_scene_images, motor_command_max_length) # call forward function

        ## Compute loss if target_caption is provided
        
        motor_command_output = motor_command_output.view(-1, motor_command_output.size(2))

        loss = self.loss_function(motor_command_output, target_caption.view(-1))
        self.log("val/loss", loss)

        return loss
    
    def predict_step(self, batch, dataloader_idx=0):
        start_scene_images, goal_scene_images, target_caption, motor_command_max_length = self.get_inference_inputs(batch)

        motor_command_output = self(start_scene_images, goal_scene_images, motor_command_max_length) # call forward function

        ## greedy decoding
        motor_command_output = motor_command_output.argmax(dim=2)

        motor_command_correct_predicted = list(zip(decode_motor_commands(target_caption), decode_motor_commands(motor_command_output)))

        accuracy = task_completion_metric(target_caption, motor_command_output)

        return motor_command_correct_predicted, accuracy

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

def get_setup(config_file, wandb_logger, checkpoint_name=None, dataloader_type='subnet2', log_and_checkpoint=True):
    
    train_config = OmegaConf.load(config_file)
    
    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    # batch_size = trial.suggest_categorical("batch_size", [16, 32])

    gpus = torch.cuda.device_count()

    if log_and_checkpoint:
        ## save checkpoints
        checkpoint_callback = instantiate_from_config(train_config.lightning.callbacks.checkpoint_callback)
        checkpoint_callback.filename = checkpoint_name
        trainer = Trainer.from_argparse_args(train_config.lightning.subnet2_trainer, gpus=gpus, logger=wandb_logger, callbacks=[checkpoint_callback])

    else:
        trainer = Trainer.from_argparse_args(train_config.lightning.subnet2_trainer, gpus=gpus)

    ## Turn this into load from config and read from lightning
    data = instantiate_from_config(train_config.data)
    data.dataloader_type = dataloader_type
    model = get_model(train_config.SubNet2)

    return trainer, model, data

def train():
    
    config_file = "/ocean/projects/cis230036p/amihretu/SmJEP/architecture/model/configs/train.yaml"
    
    ### init loggers and checkpoint callback
    run_name = datetime.now().strftime("%Y_%m_%d__%H_%M")+"SN2"#+f"__lr_{learning_rate}__bs_{batch_size}"
    checkpoint_name =  'SubNet2-'+run_name

    ## log to WANDB
    wandb_logger = WandbLogger(project='SmJEP', log_model=all, name=run_name)
    
    trainer, model, data = get_setup(config_file=config_file, wandb_logger=wandb_logger, checkpoint_name=checkpoint_name, dataloader_type='subnet2')
  
    # lr_finder = trainer.tuner.lr_find(model, datamodule=data)#, min_lr=1e-03, max_lr=1e-01)
    # import pprint
    # with open('sn2_lr_results', 'w') as f:
    #     f.write(f"{vars(lr_finder)}")
    #     f.write(f"\n\nsuggestion: {lr_finder.suggestion()}")
    # model.learning_rate = learning_rate
    # data.batch_size = batch_size
    model.train()
    trainer.fit(model, data)

    trainer.test(model, data)

def predict():

    config_file = "/ocean/projects/cis230036p/amihretu/SmJEP/architecture/model/configs/train.yaml"
    
    ### init loggers and checkpoint callback
    run_name = datetime.now().strftime("%Y_%m_%d__%H_%M")+"SN2"#+f"__lr_{learning_rate}__bs_{batch_size}"
    checkpoint_name =  'SubNet2-'+run_name

    ## log to WANDB
    wandb_logger = WandbLogger(project='SmJEP', log_model=all, name=run_name)
    
    trainer, model, data = get_setup(config_file=config_file, wandb_logger=wandb_logger, checkpoint_name=checkpoint_name, dataloader_type='subnet2', log_and_checkpoint=False)
    
    # ckpt = '/ocean/projects/cis230036p/amihretu/SmJEP/architecture/model/checkpoints/SubNet1-2023_12_18__09_57SN2.ckpt'
    # ckpt = '/ocean/projects/cis230036p/amihretu/SmJEP/architecture/model/checkpoints/SubNet1-2023_12_18__10_50SN2.ckpt'

    # model.load_from_checkpoint(ckpt)

    model.eval()

    return trainer.predict(model, data)


if __name__ == '__main__':
    train()
