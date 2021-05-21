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
        exp_name = 'dev--' + exp_name
        mode = 'local'
        nseeds = 1

    params = {
        'seed': list(range(nseeds)),
    }
    default_params = {
        'env_type': 'ant_dir',
        'offline_buffer_path': "/preloaded_data/21-02-22-ant-awac--exp7-ant-dir-4-eval-4-train-sac-to-get-buffer-longer/21-02-22-ant-awac--exp7-ant-dir-4-eval-4-train-sac-to-get-buffer-longer_2021_02_23_06_09_23_id000--s270987/extra_snapshot_itr400.cpkl",
        'saved_tasks_path': "/preloaded_data/21-05-05_pearl-awac-ant-awac--exp59-half-cheetah-130-online-pearl/16h-02m-49s_run2/tasks_description.joblib",
        'pretrained_vae_dir': "/preloaded_data/TODO/trained_vae/",
        'offline_buffer_path_to_save_to': '/preloaded_data/demos/ant_dir_4_100k_each/pearl_buffer_iter400_relabelled/',
        'transform_data_bamdp': True,
        'load_buffer_kwargs': {
            'end_idx': 200000,
            'start_idx': -100000,
        },
        'path_length': 200,
        'meta_episode_len': 600,
    }

    if mode == 'local':
        remote_mount_configs = [
            dict(
                local_dir='/home/vitchyr/mnt3/azure/',
                mount_point='/preloaded_data',
            ),
        ]
    elif mode == 'azure':
        remote_mount_configs = [
            dict(
                local_dir='/doodad_tmp/',
                mount_point='/preloaded_data',
            ),
        ]
    else:
        remote_mount_configs = []
    print(exp_name)
    sweep_function(
        borel,
        params,
        default_params=default_params,
        log_path=exp_name,
        mode=mode,
        docker_image='vitchyr/borel-v2',
        code_dirs_to_mount=[
            '/home/vitchyr/git/BOReL/',
            '/home/vitchyr/git/doodad/',
            '/home/vitchyr/git/railrl/',
            '/home/vitchyr/git/multiworld/',
        ],
        non_code_dirs_to_mount=[
            dict(
                local_dir='/home/vitchyr/.mujoco/',
                mount_point='/root/.mujoco',
            ),
        ],
        azure_mode_kwargs=dict(
            num_vcpu=16,
        ),
        use_gpu=True,
        remote_mount_configs=remote_mount_configs,
    )
    print(exp_name)


if __name__ == '__main__':
    main()
