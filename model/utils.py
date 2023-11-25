import importlib
import sys


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