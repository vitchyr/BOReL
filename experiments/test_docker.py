import click

from doodad.wrappers.easy_launch import sweep_function


def foo(*args, **kwargs):
    print("hello from test_docker")

@click.command()
@click.option('--debug', is_flag=True, default=False)
@click.option('--suffix', default=None)
@click.option('--nseeds', default=1)
@click.option('--mode', default='here_no_doodad')
def main(debug, suffix, nseeds, mode):
    path_parts = __file__.split('/')
    suffix = '' if suffix is None else '--{}'.format(suffix)
    exp_name = 'borel-{}{}'.format(
        path_parts[-1].split('.')[0],
        suffix,
    )

    if debug:
        exp_name = 'dev--' + exp_name
        mode = 'here_no_doodad'
        nseeds = 1

    params = {
        'env_type': ['hopper'],
        'seed': list(range(nseeds)),
    }
    if mode == 'azure':
        remote_mount_configs = [
            dict(
                local_dir='/doodad_tmp/21-11-14_smac-iclr22-hopper--hopper-data-gen--v4/23h-02m-27s_run0/',
                mount_point='/preloaded_data',
            ),
        ]
        exp_dir_path = '/preloaded_data/'
        use_gpu = True
    elif mode == 'here_no_doodad':
        exp_name = 'dev--' + exp_name
        remote_mount_configs = []
        exp_dir_path = '/Users/vitchyr/data/doodad/'
        use_gpu = False
    else:
        raise ValueError(mode)

    default_params = {
        'offline_buffer_path': exp_dir_path + 'extra_snapshot_itr40.cpkl',
        'saved_tasks_path': exp_dir_path + 'tasks_description.joblib',
        'vae-batch-num-rollouts-per-task': 4,
        'load_buffer_kwargs': {
            'start_idx': -1200,
            'end_idx': None,
        },
        'path_length': 200,
        'meta_episode_len': 600,
    }

    print(exp_name)
    sweep_function(
        foo,
        params,
        default_params=default_params,
        log_path=exp_name,
        mode=mode,
        docker_image='vitchyr/borel-v4',
        code_dirs_to_mount=[
            '/Users/vitchyr/code/BOReL/',
            '/Users/vitchyr/code/doodad/',
            '/Users/vitchyr/code/railrl-private/',
            '/Users/vitchyr/code/multiworld/',
            '/Users/vitchyr/code/rand_param_envs',
        ],
        use_gpu=False,
        # azure_mode_kwargs=dict(
        #     num_vcpu=16,
        #     terminate_on_end=True,
        # ),
        remote_mount_configs=remote_mount_configs,
    )
    print(exp_name)


if __name__ == '__main__':
    main()
