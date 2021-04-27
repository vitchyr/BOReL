from utils.run_experiment import offline_experiment
from doodad.wrappers.easy_launch import sweep_function


params = {'tag': ['test']}
sweep_function(
    offline_experiment,
    log_path='borel-test',
    params=params,
    docker_image='vitchyr/borel-v1',
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
    mode='local',
)
