import importlib
import sys
import torch

sys.path.append('/ocean/projects/cis230036p/amihretu/SmJEP/architecture/data')
from vocab import CRAM_MOTOR_COMMANDS, SPECIAL_TOKENS

def instantiate_from_config(config):
    ## https://github.com/timothybrooks/instruct-pix2pix
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    target = config["target"]
    package = config["path"] if "path" in config else None
    return get_obj_from_str(target, package)(**config.get("params", dict()))


def get_obj_from_str(string, path=None, reload=False):
    ## https://github.com/timothybrooks/instruct-pix2pix
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    if path is not None:
        sys.path.append(path)
    return getattr(importlib.import_module(module), cls)


def get_model(config):
    
    model = instantiate_from_config(config.model)
    
    if 'ckpt_path' in config.model.params.keys():
       checkpoint = torch.load(config.model.params.ckpt_path)
       _,_ = model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model.learning_rate = config.model.base_learning_rate
    
    return model

def decode_motor_commands(batch):
    vocabulary = CRAM_MOTOR_COMMANDS + SPECIAL_TOKENS
    # return [[(vocabulary[i]) for i in datapoint] for datapoint in batch]
    return [" ".join([vocabulary[i] for i in datapoint]) for datapoint in batch]

def task_completion_metric(correct, predicted):
    # correct_count = 0
    # for (correct, predicted) in correct_predict_list:
    #     if correct == predicted:
    #         correct_count += 1

    # return correct_count / len(correct_predict_list)
    print(correct, predicted)
    return (correct == predicted).sum() / len(predicted)
