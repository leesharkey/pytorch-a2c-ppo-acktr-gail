import glob
import os
import torch
import torch.nn as nn
from a2c_ppo_acktr.envs import VecNormalize
import pandas as pd
import os
from datetime import datetime

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def datetimenow(subseconds=False):
    if not subseconds:
        now = datetime.now()
        return now.strftime("%Y%m%d-%H%M%S")
    else:
        return datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')

def save_configs_to_csv(args, session_name=None, model_name=None,
                        unique_id=None, results_dict=None):
    """"""
    arg_dict = vars(args).copy()
    arg_dict = {**arg_dict, **{'unique_id': unique_id,
                               'model_name': model_name,
                               'session_name': session_name}}

    # Combine dicts
    if results_dict is not None:
        arg_dict = {**arg_dict, **results_dict}

    # Convert any lists into string so csvs can take them
    for k, v in arg_dict.items():
        arg_dict[k] = str(v) if type(v) ==list or type(v) ==dict else v

    # Check there isn't already a df for this unique_id; adjust model name if so
    # if os.path.isfile("exps/params_and_results_%s.csv" % unique_id) and loading:
    #     model_name = model_name + 'loaded_' + session_name

    # Create a df with a single row with new info
    full_df = pd.DataFrame(arg_dict, index=[unique_id])

    # Create a new csv if one doesn't already exist and save the new data
    params_dir = "../exps/params_and_results"
    if not os.path.isdir(params_dir):
        os.mkdir(params_dir)
    save_filename = "%s.csv" % unique_id
    full_df.to_csv(os.path.join(params_dir, save_filename))
    print("Saved params and results csv for %s." % unique_id)

def combine_all_csvs(directory_str, base_csv_name='params_and_results.csv',
                     remove_old_csvs=False):
    """Run this using python"""
    # Define some fixed variables
    archive_dir = 'archive_csvs'
    archive_path = os.path.join(directory_str, archive_dir)
    dfs = []
    files = [f for f in os.listdir(directory_str) if
             os.path.isfile(os.path.join(directory_str, f))]

    # Make archive file if keeping old csvs and it doesn't already exist
    if remove_old_csvs and not os.path.isdir(archive_path):
        os.mkdir(archive_path)

    # Loop through files, checking if they're a csv, and adding their pandas
    # dataframe to the list of dfs
    full_df = None
    for f in files:
        ext_filename = os.path.join(directory_str, f) #extended_filename
        print(ext_filename)
        if f.endswith('.csv'):
            new_df = pd.read_csv(ext_filename, header=0, index_col=0)
            dfs += [new_df]
            if full_df is not None:
                full_df = pd.concat([full_df, new_df], axis=0, sort=True)
            else:
                full_df = new_df
            if remove_old_csvs:
                os.rename(ext_filename, os.path.join(archive_path, f))

    # Concatenate all the dfs together, remove any duplicate rows, and save
    if dfs==[]:
        raise FileNotFoundError("No CSVs in the list to be merged. Check" +
                                " that your path names are correct and that" +
                                " the folder contains CSVs to merge.")
    full_df = full_df.drop_duplicates()
    full_df.to_csv(os.path.join(directory_str, base_csv_name))
