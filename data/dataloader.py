from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from vocab import ALFRED_action_dict, CRAM_MOTOR_COMMANDS, SPECIAL_TOKENS
import ast
import pandas as pd
import numpy as np

import cv2
from PIL import Image

from einops import rearrange

import os
import pandas as pd
from pprint import pprint
import sys
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule

sys.path.append("/ocean/projects/cis230036p/amihretu/SmJEP/architecture/model")
from utils import instantiate_from_config


class CRAMDataModule(LightningDataModule):
    def __init__(self, train=None, validation=None, test=None, predict=None, batch_size=2, num_workers=None, dataloader_type='subnet1'):
        super().__init__()
        self.num_workers = num_workers if num_workers is not None else 2
        self.batch_size = batch_size

        self.dataloader_type = dataloader_type

        self.train = train
        self.validation = validation
        self.test = test
        self.predict = predict
        

    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = instantiate_from_config(self.train)
            self.train_dataset.dataloader_type = self.dataloader_type
            
            self.validation_dataset = instantiate_from_config(self.validation)
            self.validation_dataset.dataloader_type = self.dataloader_type
        
        if stage == 'test':
            self.test_dataset = instantiate_from_config(self.test)
            self.test_dataset.dataloader_type = self.dataloader_type

        if stage == 'predict':
            self.prediction_dataset = instantiate_from_config(self.predict)
            self.prediction_dataset.dataloader_type = self.dataloader_type

    ## relies on the prams being present in config file -- update
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.train_dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.validation_dataset.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, collate_fn=self.test_dataset.collate_fn)
    
    def predict_dataloader(self):
        return DataLoader(self.prediction_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False, collate_fn=self.prediction_dataset.collate_fn)
    

class CRAMDataset(Dataset):
    def __init__(
        self,
        csv,
        dataset_directory,
        dataloader_type='subnet1',
        resize_resolution = 256,
        df: pd.DataFrame = None
    ):
        self.dataloader_type = dataloader_type

        if df is not None:
            self.dataset_points = df.copy()
        else:
            self.dataset_points = pd.read_csv(csv)

        self.dataset_directory = dataset_directory
        self.resize_resolution = resize_resolution

        self.motor_commands_vocabulary = CRAM_MOTOR_COMMANDS + SPECIAL_TOKENS

    def __len__(self):
        return len(self.dataset_points)

    def __getitem__(self, idx):
        
        data_point = self.dataset_points.sample_ID[idx]
        dataset_version = self.dataset_points.version[idx]
        # in state
        in_state = str(self.dataset_points.in_state[idx]) 
        in_state += ".png" if dataset_version == 'v2' else ""
        
        input_state_image = Image.open(os.path.join(
            self.dataset_directory, str(dataset_version), str(data_point), in_state
        ))
        reize_res = torch.randint(self.resize_resolution, self.resize_resolution + 1, ()).item()

        input_state_image = input_state_image.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        input_state_image = rearrange(2 * torch.tensor(np.array(input_state_image)).float() / 255 - 1, "h w c -> c h w")

        # out state
        goal_state = str(self.dataset_points.goal_state[idx])
        goal_state += ".png" if dataset_version == 'v2' else ""

        goal_state_image = Image.open(os.path.join(
            self.dataset_directory, str(dataset_version), str(data_point), goal_state
        ))
        goal_state_image = goal_state_image.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        goal_state_image = rearrange(2 * torch.tensor(np.array(goal_state_image)).float() / 255 - 1, "h w c -> c h w")
        
        if self.dataloader_type == 'subnet1':
            # action desc
            action_description = self.dataset_points.action_description[idx]
            
            return dict(edited=goal_state_image, edit=dict(c_concat=input_state_image, c_crossattn=action_description))

        elif self.dataloader_type == 'subnet2':
            ## motor cmd
            motor_command = self.dataset_points.motor_cmd[idx].split()
            motor_command = [self.motor_commands_vocabulary.index(
                token) for token in motor_command]
            motor_command = torch.tensor(motor_command)
            motor_command = motor_command

            return dict(start_state_image=input_state_image, goal_state_image=goal_state_image, motor_command=motor_command)
        

    def collate_fn(self, batch):

        if self.dataloader_type == 'subnet1':
            return torch.utils.data.dataloader.default_collate(batch)

        elif self.dataloader_type == 'subnet2':
            batch_input_state = torch.stack([item['start_state_image'] for item in batch])
            batch_goal_state = torch.stack([item['goal_state_image'] for item in batch])
            batch_motor_commands = [item['motor_command'] for item in batch]
            
            padding_value = self.motor_commands_vocabulary.index("[PAD]")

            batch_motor_commands = pad_sequence(
                batch_motor_commands, batch_first=True, padding_value=padding_value)

            return dict(start_state_image=batch_input_state, goal_state_image=batch_goal_state, motor_command=batch_motor_commands)
              

class ALFRED_Dataset(Dataset):
    def __init__(self, csv_file):
        self.csv = pd.read_csv(csv_file)
        self.csv['sub_goal'] = self.csv['sub_goal'].apply(lambda r: ast.literal_eval(r))
        self.dir = "/ocean/projects/cis230036p/amihretu/SmJEP/dataset/ALFRED/full_2.1.0/"

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        current_datapoint = self.csv.iloc[idx]
        image_directory = self.dir+"/"+current_datapoint['data_folder']+"/"+current_datapoint['task_name']+"/"+current_datapoint['trial_id']+"/raw_images/"
        
        task_description = current_datapoint['task_description']
        sub_goals = current_datapoint['sub_goal']
        start_images = [cv2.imread(image_directory+image_set[0].split(".")[0]+".jpg") for sub_goal in sub_goals for image_set in sub_goal['images'] if sub_goal['images'] != []] # select only one?
        goal_images = [cv2.imread(image_directory+image_set[1].split(".")[0]+".jpg") for sub_goal in sub_goals for image_set in sub_goal['images'] if sub_goal['images'] != []]
        
        # start_image = Image.open(image_directory+image_set[0].split(".")[0]+".jpg") sub_goals[0] if sub_goals[0]['images'] != []
        # goal_image = Image.read(image_directory+image_set[0].split(".")[0]+".jpg") sub_goals[0] if sub_goals[0]['images'] != []
        
        reize_res = torch.randint(256, 256 + 1, ()).item()
        start_image = start_image.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        actions = [self.get_actions(sub_goal['low_level_actions']) for sub_goal in sub_goals if sub_goal['low_level_actions'] != []]
        actions = pad_sequence(actions, padding_value=-1, batch_first=True)

        return task_description, start_images, goal_images, actions

    @staticmethod
    def get_actions(list_of_actions):
        list_of_indices = [ALFRED_action_dict[action_class] for action_class in list_of_actions]
        return torch.tensor(list_of_indices)
    

def get_ALFRED_DataLoader(csv_file, batch_size=1, shuffle=True):
    
    dataset = ALFRED_Dataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    return dataloader
