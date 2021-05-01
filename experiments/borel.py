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
    exp_name = 'borel-{}{}'.format(
        path_parts[-1].split('.')[0],
        suffix,
    )

    if debug:
        exp_name = 'dev--' + exp_name
        mode = 'local'
        nseeds = 1

    params = {
        'env_type': ['ant_dir'],
        'seed': list(range(nseeds)),
    }
    default_params = {
        'offline_buffer_path': '/preloaded_buffer/ant_dir_32/borel_buffer_iter50_relabelled/',
        'pretrained_vae_dir': '/pretrained_vae',
        'vae_model_name': 'relabel_semi_modified_init_state_dist__30_04_23_21_33',
        'transform_data_bamdp': False,
        # 'save_data_path': '/preloaded_buffer/ant_dir_32/borel_buffer_iter50_relabelled/',
        # 'transform_data_bamdp': True,
        # 'train_task_idxs': [0, 1, 2, 3],
        # 'test_task_idxs': [4, 5, 6, 7],
    }

    if mode == 'local':
        remote_mount_configs = [
            dict(
                local_dir='/home/vitchyr/mnt2/log2/demos/',
                mount_point='/preloaded_buffer',
            ),
            dict(
                local_dir='/home/vitchyr/mnt2/log2/21-04-30_macaw-train_vae/run0_16h-19m-41s/trained_vae/',
                mount_point='/pretrained_vae',
            ),
        ]
    elif mode == 'azure':
        remote_mount_configs = [
            dict(
                local_dir='/doodad_tmp/demos/',
                mount_point='/preloaded_buffer',
            ),
        ]
    else:
        remote_mount_configs = []
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
        ],
        non_code_dirs_to_mount=[
            dict(
                local_dir='/home/vitchyr/.mujoco/',
                mount_point='/root/.mujoco',
            ),
        ],
        use_gpu=True,
        remote_mount_configs=remote_mount_configs,
    )


if __name__ == '__main__':
    main()
