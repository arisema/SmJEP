from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from vocab import ALFRED_action_dict
import ast
import pandas as pd
import numpy as np

import cv2
from PIL import Image

from einops import rearrange

import os
import os.path as osp
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
    def __init__(self, train=None, validation=None, test_KAKO=None, test_KAUO=None, test_UAKO=None, test_UAUO=None, batch_size=2, num_workers=None):
        super().__init__()
        self.num_workers = num_workers if num_workers is not None else 2
        self.batch_size = batch_size

        self.train = train
        self.validation = validation
        self.test_KAKO = test_KAKO
        self.test_KAUO = test_KAUO
        self.test_UAKO = test_UAKO
        self.test_UAUO = test_UAUO
        

    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = instantiate_from_config(self.train)
            self.validation_dataset = instantiate_from_config(self.validation)
        if stage == 'test':
            self.test_dataset_KAKO = instantiate_from_config(self.test_KAKO)
            self.test_dataset_KAUO = instantiate_from_config(self.test_KAUO)
            self.test_dataset_UAKO = instantiate_from_config(self.test_UAKO)
            self.test_dataset_UAUO = instantiate_from_config(self.test_UAUO)

        # if self.train is not None:
        #     self.train_dataset = instantiate_from_config(self.train)
        # if self.validation is not None:
        #     self.validation_dataset = instantiate_from_config(self.validation)
        # if self.test_KAKO is not None:
        #     self.test_dataset_KAKO = instantiate_from_config(self.test_KAKO)
        # if self.test_KAUO is not None:
        #     self.test_dataset_KAUO = instantiate_from_config(self.test_KAUO)
        # if self.test_UAKO is not None:
        #     self.test_dataset_UAKO = instantiate_from_config(self.test_UAKO)
        # if self.test_UAUO is not None:
        #     self.test_dataset_UAUO = instantiate_from_config(self.test_UAUO)

    ## relies on the prams being present in config file -- update
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return [DataLoader(self.test_dataset_KAKO, num_workers=self.num_workers), 
                DataLoader(self.test_dataset_KAUO, num_workers=self.num_workers), 
                DataLoader(self.test_dataset_UAKO, num_workers=self.num_workers), 
                DataLoader(self.test_dataset_UAUO, num_workers=self.num_workers)]
    

class CRAMDataset(Dataset):
    def __init__(
        self,
        csv,
        dataset_directory,
        resize_resolution = 256,
        df: pd.DataFrame = None
    ):

        if df is not None:
            self.dataset_points = df.copy()
        else:
            self.dataset_points = pd.read_csv(csv)

        self.dataset_directory = dataset_directory
        self.resize_resolution = resize_resolution

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

        # action desc
        action_description = self.dataset_points.action_description[idx]
        # action_description = tokenize_by_space(
        #     self.dataset_points.action_description[idx]
        # )
        # action_description = [self.action_description_vocabulary.index(
        #     token) for token in action_description]
        # action_description = torch.tensor(action_description)

        # # motor cmd
        # motor_command = tokenize_by_space(
        #     self.dataset_points.motor_cmd[idx])
        # motor_command = [self.motor_commands_vocabulary.index(
        #     token) for token in motor_command]
        # motor_command = torch.tensor(motor_command)

        return dict(edited=goal_state_image, edit=dict(c_concat=input_state_image, c_crossattn=action_description))

    # def collate_fn(self, batch):
    #     batch_input_state = [input_state for input_state, _, _, _ in batch]
    #     batch_goal_state = [goal_state for _, goal_state, _, _ in batch]
    #     batch_action_description = [ad for _, _, ad, _ in batch]
    #     batch_motor_commands = [motor_cmd for _, _, _, motor_cmd in batch]

    #     batch_action_description = pad_sequence(
    #         batch_action_description, padding_value=-1)

    #     batch_motor_commands = pad_sequence(
    #         batch_motor_commands, padding_value=-1)

    #     return torch.as_tensor(batch_input_state), torch.as_tensor(batch_goal_state), torch.as_tensor(batch_action_description), torch.as_tensor(batch_motor_commands)


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
