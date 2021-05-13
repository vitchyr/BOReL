import click

from doodad.wrappers.easy_launch import sweep_function
from launcher.train_vae_offline import train_vae


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
        'env_type': ['cheetah_vel'],
        'seed': list(range(nseeds)),
    }
    default_params = {
        'pretrain_buffer_path': "/preloaded_data/21-05-05_pearl-awac-ant-awac--exp59-half-cheetah-130-online-pearl/16h-02m-49s_run2/extra_snapshot_itr50.cpkl",
        'saved_tasks_path': "/preloaded_data/21-05-05_pearl-awac-ant-awac--exp59-half-cheetah-130-online-pearl/16h-02m-49s_run2/tasks_description.joblib",
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
        train_vae,
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
        use_gpu=True,
        remote_mount_configs=remote_mount_configs,
    )
    print(exp_name)


if __name__ == '__main__':
    main()
