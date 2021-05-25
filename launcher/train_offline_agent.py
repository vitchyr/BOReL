
import joblib
import random
import numpy as np
import os
import argparse
import torch
from torchkit.pytorch_utils import set_gpu_mode
from models.vae import VAE
from offline_metalearner import OfflineMetaLearner
import utils.config_utils as config_utl
from utils import offline_utils as off_utl
from offline_config import (
    args_ant_semicircle_sparse,
    args_cheetah_vel, args_point_robot_sparse, args_gridworld,
    args_ant_dir
)
from utils.logging import setup_logger, logger

env_name_to_args = {
    'ant_dir': args_ant_dir,
    'cheetah_vel': args_cheetah_vel,
}

from doodad.wrappers.easy_launch import save_doodad_config, DoodadConfig


def _borel(
        log_dir,
        pretrained_vae_dir,
        env_type,
        transform_data_bamdp,
        seed,
        path_length,
        meta_episode_len,
        relabelled_data_dir=None,
        offline_buffer_path_to_save_to=None,
        offline_buffer_path='',
        saved_tasks_path='',
        debug=False,
        vae_model_name=None,
        load_buffer_kwargs=None,
        **kwargs,
):
    if load_buffer_kwargs is None:
        load_buffer_kwargs = {}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    parser = argparse.ArgumentParser()

    if offline_buffer_path_to_save_to is None:
        offline_buffer_path_to_save_to = os.path.join(log_dir, 'transformed_data')

    # parser.add_argument('--env-type', default='gridworld')
    # parser.add_argument('--env-type', default='point_robot_sparse')
    # parser.add_argument('--env-type', default='cheetah_vel')
    parser.add_argument('--env-type', default=env_type)
    extra_args = []
    for k, v in kwargs.items():
        extra_args.append('--{}'.format(k))
        extra_args.append(str(v))
    args, rest_args = parser.parse_known_args(args=extra_args)
    args = env_name_to_args[env_type].get_args(rest_args)
    set_gpu_mode(torch.cuda.is_available() and args.use_gpu)

    if vae_model_name is None:
        vae_model_name = os.listdir(
            os.path.join(pretrained_vae_dir, args.env_name)
        )[0]

    vae_args = config_utl.load_config_file(os.path.join(pretrained_vae_dir, args.env_name,
                                                        vae_model_name, 'online_config.json'))
    args = config_utl.merge_configs(vae_args, args)     # order of input to this function is important
    # _, env = off_utl.expand_args(args)
    from environments.make_env import make_env
    task_data = joblib.load(saved_tasks_path)
    tasks = task_data['tasks']
    args.presampled_tasks = tasks
    env = make_env(args.env_name,
                   args.max_rollouts_per_task,
                   presampled_tasks=tasks,
                   seed=args.seed,
                   n_tasks=1)

    args.vae_dir = pretrained_vae_dir
    args.data_dir = None
    args.vae_model_name = vae_model_name
    if transform_data_bamdp:
        # Transform data BAMDP (state relabelling)
        # load VAE for state relabelling
        print("performing state-relabeling")
        vae_models_path = os.path.join(pretrained_vae_dir, args.env_name,
                                       vae_model_name, 'models')
        vae = VAE(args)
        off_utl.load_trained_vae(vae, vae_models_path)
        # load data and relabel
        os.makedirs(offline_buffer_path_to_save_to, exist_ok=True)
        dataset, goals = off_utl.load_pearl_buffer(
            offline_buffer_path,
            tasks,
            add_done_info=env.add_done_info,
            path_length=path_length,
            meta_episode_len=meta_episode_len,
            **load_buffer_kwargs
        )
        dataset = [[x.astype(np.float32) for x in d] for d in dataset]
        bamdp_dataset = off_utl.transform_mdps_ds_to_bamdp_ds(dataset, vae, args)
        # save relabelled data
        print("saving state-relabeled data to ", offline_buffer_path_to_save_to)
        off_utl.save_dataset(offline_buffer_path_to_save_to, bamdp_dataset, goals)
        relabelled_data_dir = offline_buffer_path_to_save_to
    args.relabelled_data_dir = relabelled_data_dir
    args.max_rollouts_per_task = 3
    args.results_log_dir = log_dir

    if debug:
        print("DEBUG MODE ON")
        args.rl_updates_per_iter = 1
        args.log_interval = 1
    learner = OfflineMetaLearner(args)

    learner.train()


def borel(doodad_config: DoodadConfig, params):
    save_doodad_config(doodad_config)
    log_dir = doodad_config.output_directory
    exp_name = log_dir.split('/')[-2]
    setup_logger(logger, variant=params, base_log_dir=None, exp_name=exp_name, log_dir=log_dir)
    _borel(log_dir, **params)
