import click

from doodad.wrappers.easy_launch import sweep_function
from launcher.train_offline_agent import borel


@click.command()
@click.option('--debug', is_flag=True, default=False)
@click.option('--suffix', default=None)
@click.option('--nseeds', default=1)
@click.option('--mode', default='local')
def main(debug, suffix, nseeds, mode):
    path_parts = __file__.split('/')
    suffix = '' if suffix is None else '--{}'.format(suffix)
    exp_name = 'borel--{}-{}{}'.format(
        path_parts[-2].replace('_', '-'),
        path_parts[-1].split('.')[0],
        suffix,
    )

    if debug:
        mode = 'here_no_doodad'
        nseeds = 1


    if mode == 'azure':
        remote_mount_configs = [
            dict(
                local_dir='/doodad_tmp/21-11-14_smac-iclr22-walker--walker-data-gen--v4/23h-02m-26s_run0/',
                mount_point='/preloaded_data',
            ),
            dict(
                local_dir='/doodad_tmp/21-12-28_borel-exp1_vae_walker/run0_16h-39m-18s/trained_vae',
                mount_point='/preloaded_vae',
            ),
        ]
        exp_dir_path = '/preloaded_data/'
        use_gpu = True
        pretrained_vae_dir = "/preloaded_vae"
    elif mode == 'here_no_doodad':
        exp_name = 'dev--' + exp_name
        remote_mount_configs = []
        exp_dir_path = '/Users/vitchyr/data/doodad/21-11-14_smac-iclr22-walker--walker-data-gen--v4/23h-02m-26s_run0/'
        use_gpu = False
        pretrained_vae_dir = "/Users/vitchyr/data/doodad/21-12-28_borel-exp1_vae_walker/run0_16h-39m-18s/trained_vae"
    else:
        raise ValueError(mode)

    params = {
        'env_type': ['walker'],
        'seed': list(range(nseeds)),
    }
    default_params = {
        'offline_buffer_path': exp_dir_path + 'extra_snapshot_itr40.cpkl',
        'saved_tasks_path': exp_dir_path + 'tasks_description.joblib',
        'vae_model_name': 'relabel__29_12_00_53_20',
        'pretrained_vae_dir': pretrained_vae_dir,
        # 'relabelled_data_dir': '/preloaded_data/demos/half_cheetah_vel_130/pearl_buffer_iter50_relabelled_v2/',
        'transform_data_bamdp': True,
        'load_buffer_kwargs': {
            'start_idx': -1200,
            'end_idx': None,
        },
        'path_length': 200,
        'meta_episode_len': 600,
        'num-train-tasks': 100,
        # 'gpu_id': 1,
    }
    print(exp_name)
    sweep_function(
        borel,
        params,
        default_params=default_params,
        log_path=exp_name,
        mode=mode,
        docker_image='vitchyr/borel-v5',
        code_dirs_to_mount=[
            '/Users/vitchyr/code/BOReL/',
            '/Users/vitchyr/code/doodad/',
            '/Users/vitchyr/code/railrl-private/',
            '/Users/vitchyr/code/multiworld/',
            '/Users/vitchyr/code/rand_param_envs',
        ],
        azure_mode_kwargs=dict(
            num_vcpu=16,
        ),
        use_gpu=use_gpu,
        remote_mount_configs=remote_mount_configs,
    )
    print(exp_name)


if __name__ == '__main__':
    main()
