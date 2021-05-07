
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
}

from doodad.wrappers.easy_launch import save_doodad_config, DoodadConfig


def _borel(
        log_dir,
        offline_buffer_path,
        pretrained_vae_dir,
        vae_model_name,
        env_type,
        transform_data_bamdp,
        seed,
        save_data_path='',
        debug=False,
        max_rollouts_per_task=3,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    parser = argparse.ArgumentParser()

    # parser.add_argument('--env-type', default='gridworld')
    # parser.add_argument('--env-type', default='point_robot_sparse')
    # parser.add_argument('--env-type', default='cheetah_vel')
    parser.add_argument('--env-type', default=env_type)
    args, rest_args = parser.parse_known_args(args=[])
    args = env_name_to_args[env_type].get_args(rest_args)
    set_gpu_mode(torch.cuda.is_available() and args.use_gpu)

    vae_args = config_utl.load_config_file(os.path.join(pretrained_vae_dir, args.env_name,
                                                        vae_model_name, 'online_config.json'))
    args = config_utl.merge_configs(vae_args, args)     # order of input to this function is important
    # _, env = off_utl.expand_args(args)
    from environments.make_env import make_env
    env = make_env(args.env_name,
                   args.max_rollouts_per_task,
                   seed=args.seed,
                   n_tasks=1)

    # Transform data BAMDP (state relabelling)
    args.vae_dir = pretrained_vae_dir
    args.data_dir = offline_buffer_path
    if transform_data_bamdp:
        # load VAE for state relabelling
        vae_models_path = os.path.join(pretrained_vae_dir, args.env_name,
                                       vae_model_name, 'models')
        vae = VAE(args)
        off_utl.load_trained_vae(vae, vae_models_path)
        # load data and relabel
        os.makedirs(save_data_path, exist_ok=True)
        # dataset, goals = off_utl.load_dataset(data_dir=args.data_dir, args=args, arr_type='numpy')
        dataset, goals = off_utl.load_rlkit_to_macaw_dataset(
            data_dir=offline_buffer_path,
            add_done_info=env.add_done_info,
        )
        dataset = [[x.astype(np.float32) for x in d] for d in dataset]
        bamdp_dataset = off_utl.transform_mdps_ds_to_bamdp_ds(dataset, vae, args)
        # save relabelled data
        off_utl.save_dataset(save_data_path, bamdp_dataset, goals)
        args.relabelled_data_dir = save_data_path
    else:
        args.relabelled_data_dir = offline_buffer_path

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
