from pathlib import Path
import re
import string


def get_epochs_with_best(cfg_use_epochs, some_asset, namer):
    """
    Get a list of epochs to use for training or prediction, including 'best' epochs if specified.
    Note these assumptions:
    - specified epochs are integers or "best"
    - integer epochs are assumed to exist
    - "best" epochs are determined by the presence of files named "model_best_<epoch>.pth" in the model asset directory
    - if a best epoch exists, the normal epoch will not
    """
    use_epoch_nums = [e for e in cfg_use_epochs if e != "best"]

    if "best" in cfg_use_epochs:
        best_epoch_nums = []

        # Open up the Namer; not recommended.
        known_context = namer.context
        known_context["stage"] = some_asset.source_stage_dir
        path_template = some_asset.path_template

        template_tags = re.findall(r'{(.*?)}', path_template)
        temp_context = {}
        for tag in template_tags:
            if tag in known_context.keys():
                continue
            if tag in ["epoch"]:
                continue
            current_context = {**known_context, **temp_context}
            formatter = SafeFormatter(current_context)
            current_path = formatter.format(path_template)
            current_path = current_path[:current_path.find("{")]
            # Get all subfolders of current path
            current_path = Path(current_path)
            sub_folders = get_subfolders(current_path)
            if not sub_folders:
                raise ValueError(f"No subfolders found in {current_path}.")
            temp_context[tag] = sub_folders[0].name

        temp_context["epoch"] = "$"

        with namer.set_contexts(temp_context):
            model_asset_path = some_asset.path

        # get all model files in parent directory starting with "model_best_"
        best_model_files = model_asset_path.parent.glob("*best_*.*")
        for path_template in best_model_files:
            epoch_num = int(path_template.name.split('_')[-1].split('.')[0])
            best_epoch_nums.append(epoch_num)

        use_epoch_nums.extend(best_epoch_nums)
        use_epoch_nums = sorted(set(use_epoch_nums))

        res_epochs = []
        for e in use_epoch_nums:
            if e in best_epoch_nums:
                res_epochs.append(f"best_{e}")
            else:
                res_epochs.append(e)
    else:
        res_epochs = use_epoch_nums

    return res_epochs


class SafeFormatter(string.Formatter):
    def __init__(self, known_values):
        self.known_values = known_values

    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return self.known_values.get(key, f'{{{key}}}')
        return super().get_value(key, args, kwargs)


def get_subfolders(path: Path):
    return [p for p in path.iterdir() if p.is_dir() and p.name != "Logs"]
