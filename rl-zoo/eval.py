import argparse
import difflib
import os
from collections import OrderedDict
from pprint import pprint
import warnings
import importlib

# For pybullet envs
warnings.filterwarnings("ignore")
import gym
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
import numpy as np
import yaml
try:
    import highway_env
except ImportError:
    highway_env = None
from mpi4py import MPI

from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.ppo2.ppo2 import constfn

from utils import make_env, ALGOS, linear_schedule, get_latest_run_id, get_wrapper_class
from utils.hyperparams_opt import hyperparam_optimization
from utils.noise import LinearNormalActionNoise
import utils
import sys
import tensorflow as tf
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from skimage.transform import resize
from skimage.util import img_as_ubyte

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, nargs='+', default=["CartPole-v1"], help='environment ID(s)')
    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
    parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                        default='', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
                        type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,
                        type=int)
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--n-trials', help='Number of trials for optimizing hyperparameters', type=int, default=10)
    parser.add_argument('-optimize', '--optimize-hyperparameters', action='store_true', default=False,
                        help='Run hyperparameters search')
    parser.add_argument('--n-jobs', help='Number of parallel jobs when optimizing hyperparameters', type=int, default=1)
    parser.add_argument('--sampler', help='Sampler to use when optimizing hyperparameters', type=str,
                        default='tpe', choices=['random', 'tpe', 'skopt'])
    parser.add_argument('--pruner', help='Pruner to use when optimizing hyperparameters', type=str,
                        default='median', choices=['halving', 'median', 'none'])
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    parser.add_argument('--trained-agent-folder', help='Path to a pretrained agent to demo training results', default='', type=str)

    # subpolicy hypers
    parser.add_argument('--policy-cost-coef', type=float, default=2.9e-2)
    parser.add_argument("--sub-policy-costs", nargs="*", type=float, default=[1, 20])
    parser.add_argument("--sub-hidden-sizes", nargs="*", type=int, default=[8, 64])

    # need to record frames for the last few episodes?
    parser.add_argument('--n-episodes-record-frames', help='Number of episodes to record frames', type=int, default=8)
    parser.add_argument('--n-eval-episodes', help='Number of evaluation episodes', type=int, default=500)

    args = parser.parse_args()

    assert args.trained_agent_folder != ''

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_ids = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())

    for env_id in env_ids:
        # If the environment is not found, suggest the closest match
        if env_id not in registered_envs:
            try:
                closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
            except IndexError:
                closest_match = "'no close match found...'"
            raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

    set_global_seeds(args.seed)

    if args.trained_agent != "":
        valid_extension = args.trained_agent.endswith('.pkl') or args.trained_agent.endswith('.zip')
        assert valid_extension and os.path.isfile(args.trained_agent), \
            "The trained_agent must be a valid path to a .zip/.pkl file"

    rank = 0
    if MPI.COMM_WORLD.Get_size() > 1:
        print("Using MPI for multiprocessing with {} workers".format(MPI.COMM_WORLD.Get_size()))
        rank = MPI.COMM_WORLD.Get_rank()
        print("Worker rank: {}".format(rank))

        args.seed += rank
        if rank != 0:
            args.verbose = 0
            args.tensorboard_log = ''

    for env_id in env_ids:
        tensorboard_log = None if args.tensorboard_log == '' else os.path.join(args.tensorboard_log, env_id)

        is_atari = False
        if 'NoFrameskip' in env_id:
            is_atari = True

        print("=" * 10, env_id, "=" * 10)

        # Load hyperparameters from yaml file
        # with open('hyperparams/{}.yml'.format(args.algo), 'r') as f:
        if 'Fetch' in args.env[0]:    # or args.algo == "her"?
            hyper_file = 'hyperparams/her-dqn.yml'
        else:
            hyper_file = 'hyperparams/{}.yml'.format(args.algo)
        with open(hyper_file, 'r') as f:
            hyperparams_dict = yaml.load(f)
            if env_id in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[env_id]
            elif is_atari:
                hyperparams = hyperparams_dict['atari']
            else:
                raise ValueError("Hyperparameters not found for {}-{}".format(args.algo, env_id))

        # Sort hyperparams that will be saved
        saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

        algo_ = args.algo
        # HER is only a wrapper around an algo
        if args.algo == 'her':
            algo_ = saved_hyperparams['model_class']
            assert algo_ in {'sac', 'ddpg', 'dqn', 'td3'}, "{} is not compatible with HER".format(algo_)
            # Retrieve the model class
            hyperparams['model_class'] = ALGOS[saved_hyperparams['model_class']]

        if args.verbose > 0:
            pprint(saved_hyperparams)

        n_envs = hyperparams.get('n_envs', 1)

        if args.verbose > 0:
            print("Using {} environments".format(n_envs))

        # Create learning rate schedules for ppo2 and sac
        if algo_ in ["ppo2", "sac", "td3"]:
            for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
                if key not in hyperparams:
                    continue
                if isinstance(hyperparams[key], str):
                    schedule, initial_value = hyperparams[key].split('_')
                    initial_value = float(initial_value)
                    hyperparams[key] = linear_schedule(initial_value)
                elif isinstance(hyperparams[key], (float, int)):
                    # Negative value: ignore (ex: for clipping)
                    if hyperparams[key] < 0:
                        continue
                    hyperparams[key] = constfn(float(hyperparams[key]))
                else:
                    raise ValueError('Invalid value for {}: {}'.format(key, hyperparams[key]))

        # Should we overwrite the number of timesteps?
        if args.n_timesteps > 0:
            if args.verbose:
                print("Overwriting n_timesteps with n={}".format(args.n_timesteps))
            n_timesteps = args.n_timesteps
        else:
            n_timesteps = int(hyperparams['n_timesteps'])

        normalize = False
        normalize_kwargs = {}
        if 'normalize' in hyperparams.keys():
            normalize = hyperparams['normalize']
            if isinstance(normalize, str):
                normalize_kwargs = eval(normalize)
                normalize = True
            del hyperparams['normalize']

        if 'policy_kwargs' in hyperparams.keys():
            hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])

        # Delete keys so the dict can be pass to the model constructor
        if 'n_envs' in hyperparams.keys():
            del hyperparams['n_envs']
        del hyperparams['n_timesteps']

        # obtain a class object from a wrapper name string in hyperparams
        # and delete the entry
        if 'Fetch' in args.env[0]:
            env_wrapper = get_wrapper_class({'env_wrapper': 'utils.wrappers.DoneOnSuccessWrapper'})
        else:
            env_wrapper = None
        if 'env_wrapper' in hyperparams.keys():
            del hyperparams['env_wrapper']

        def create_env(n_envs):
            """
            Create the environment and wrap it if necessary
            :param n_envs: (int)
            :return: (gym.Env)
            """
            global hyperparams

            if is_atari:
                if args.verbose > 0:
                    print("Using Atari wrapper")
                env = make_atari_env(env_id, num_env=n_envs, seed=args.seed)
                # Frame-stacking with 4 frames
                env = VecFrameStack(env, n_stack=4)
            elif algo_ in ['dqn', 'ddpg']:
                if hyperparams.get('normalize', False):
                    print("WARNING: normalization not supported yet for DDPG/DQN")
                env = gym.make(env_id)
                env.seed(args.seed)
                if env_wrapper is not None:
                    env = env_wrapper(env)
            else:
                if n_envs == 1:
                    env = DummyVecEnv([make_env(env_id, 0, args.seed, wrapper_class=env_wrapper)])
                else:
                    # env = SubprocVecEnv([make_env(env_id, i, args.seed) for i in range(n_envs)])
                    # On most env, SubprocVecEnv does not help and is quite memory hungry
                    env = DummyVecEnv([make_env(env_id, i, args.seed, wrapper_class=env_wrapper) for i in range(n_envs)])
                if normalize:
                    if args.verbose > 0:
                        if len(normalize_kwargs) > 0:
                            print("Normalization activated: {}".format(normalize_kwargs))
                        else:
                            print("Normalizing input and reward")
                    env = VecNormalize(env, **normalize_kwargs)
            # Optional Frame-stacking
            if hyperparams.get('frame_stack', False):
                n_stack = hyperparams['frame_stack']
                env = VecFrameStack(env, n_stack)
                print("Stacking {} frames".format(n_stack))
                del hyperparams['frame_stack']
            return env


        env = create_env(n_envs)
        # Stop env processes to free memory
        if args.optimize_hyperparameters and n_envs > 1:
            env.close()

        # Parse noise string for DDPG and SAC
        if algo_ in ['ddpg', 'sac', 'td3'] and hyperparams.get('noise_type') is not None:
            noise_type = hyperparams['noise_type'].strip()
            noise_std = hyperparams['noise_std']
            n_actions = env.action_space.shape[0]
            if 'adaptive-param' in noise_type:
                assert algo_ == 'ddpg', 'Parameter is not supported by SAC'
                hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                                    desired_action_stddev=noise_std)
            elif 'normal' in noise_type:
                if 'lin' in noise_type:
                    hyperparams['action_noise'] = LinearNormalActionNoise(mean=np.zeros(n_actions),
                                                                          sigma=noise_std * np.ones(n_actions),
                                                                          final_sigma=hyperparams.get('noise_std_final', 0.0) * np.ones(n_actions),
                                                                          max_steps=n_timesteps)
                else:
                    hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                                    sigma=noise_std * np.ones(n_actions))
            elif 'ornstein-uhlenbeck' in noise_type:
                hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                           sigma=noise_std * np.ones(n_actions))
            else:
                raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
            print("Applying {} noise with std {}".format(noise_type, noise_std))
            del hyperparams['noise_type']
            del hyperparams['noise_std']
            if 'noise_std_final' in hyperparams:
                del hyperparams['noise_std_final']

        if args.trained_agent_folder != '':
            # Continue training
            print("Loading pretrained agent")
            # Policy should not be changed
            del hyperparams['policy']

            trained_m_path = os.path.join(args.trained_agent_folder, '%s.zip' % env_id)
            model = ALGOS[args.algo].load(trained_m_path, env=env,
                                          tensorboard_log=tensorboard_log, verbose=args.verbose, **hyperparams)

            exp_folder = args.trained_agent[:-4]
            if normalize:
                print("Loading saved running average")
                env.load_running_average(exp_folder)
            print("Macro policy loaded!!!!!!")

        elif args.optimize_hyperparameters:

            if args.verbose > 0:
                print("Optimizing hyperparameters")


            def create_model(*_args, **kwargs):
                """
                Helper to create a model with different hyperparameters
                """
                return ALGOS[args.algo](env=create_env(n_envs), tensorboard_log=tensorboard_log,
                                        verbose=0, **kwargs)


            data_frame = hyperparam_optimization(args.algo, create_model, create_env, n_trials=args.n_trials,
                                                 n_timesteps=n_timesteps, hyperparams=hyperparams,
                                                 n_jobs=args.n_jobs, seed=args.seed,
                                                 sampler_method=args.sampler, pruner_method=args.pruner,
                                                 verbose=args.verbose)

            report_name = "report_{}_{}-trials-{}-{}-{}.csv".format(env_id, args.n_trials, n_timesteps,
                                                                    args.sampler, args.pruner)

            log_path = os.path.join(args.log_folder, args.algo, report_name)

            if args.verbose:
                print("Writing report to {}".format(log_path))

            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            data_frame.to_csv(log_path)
            exit()
        else:
            # Train an agent from scratch
            model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, verbose=args.verbose, **hyperparams)

        kwargs = {}
        if args.log_interval > -1:
            kwargs = {'log_interval': args.log_interval}


        n_sub_models = 2
        sub_models = []
        replay_wrappers = []
        if args.algo == "her":
            # get model inside her wrapper
            inner_model = model.model
        else:
            inner_model = model
        with inner_model.graph.as_default():
            for i in range(n_sub_models):
                # Load hyperparameters for sub_policy from yaml file
                if 'Fetch' in args.env[0]:
                    hyper_file = 'hyperparams/her.yml'
                    sub_algo = "her"
                else:
                    hyper_file = 'hyperparams/sac.yml'
                    sub_algo = "sac"
                with open(hyper_file, 'r') as f:
                    hyperparams_dict = yaml.load(f)
                    if env_id in list(hyperparams_dict.keys()):
                        sub_hyperparams = hyperparams_dict[env_id]
                    elif is_atari:
                        # hyperparams = hyperparams_dict['atari']
                        raise ValueError("Not supporting atari envs")
                    else:
                        raise ValueError("Hyperparameters not found for {}-{}".format(hyper_file, env_id))

                # prepare hyperparams
                for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
                    if key not in sub_hyperparams:
                        continue
                    if isinstance(sub_hyperparams[key], str):
                        schedule, initial_value = sub_hyperparams[key].split('_')
                        initial_value = float(initial_value)
                        sub_hyperparams[key] = linear_schedule(initial_value)
                    elif isinstance(sub_hyperparams[key], (float, int)):
                        # Negative value: ignore (ex: for clipping)
                        if sub_hyperparams[key] < 0:
                            continue
                        sub_hyperparams[key] = constfn(float(sub_hyperparams[key]))
                    else:
                        raise ValueError('Invalid value for {}: {}'.format(key, sub_hyperparams[key]))
                sub_hyperparams['policy_kwargs'] = dict(layers=[args.sub_hidden_sizes[i]] * 2)  # 2 layers

                # Delete keys so the dict can be pass to the model constructor
                if 'n_envs' in sub_hyperparams.keys():
                    del sub_hyperparams['n_envs']
                del sub_hyperparams['n_timesteps']

                # obtain a class object from a wrapper name string in hyperparams
                # and delete the entry
                if 'env_wrapper' in sub_hyperparams.keys():
                    del sub_hyperparams['env_wrapper']

                if 'model_class' in sub_hyperparams:
                    sub_hyperparams['model_class'] = ALGOS["sac"]

                if 'noise_type' in sub_hyperparams and 'ornstein-uhlenbeck' == sub_hyperparams['noise_type']:
                    n_actions = env.action_space.shape[0]
                    noise_std = sub_hyperparams['noise_std']
                    hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                       sigma=noise_std * np.ones(n_actions))
                    del sub_hyperparams['noise_type']
                    del sub_hyperparams['noise_std']

                if args.trained_agent_folder == '':
                    sub_model = ALGOS[sub_algo](env=env, verbose=args.verbose, **sub_hyperparams)
                else:
                    del sub_hyperparams['policy']
                    trained_m_path = os.path.join(args.trained_agent_folder, '%s_sub%d.zip' % (env_id, i))
                    sub_model = ALGOS[sub_algo].load(trained_m_path, env=env,
                                                     verbose=args.verbose, **sub_hyperparams)
                    print("Subpolicy %d loaded from pretrained" % i)

                if sub_algo == "her":
                    replay_wrappers.append(sub_model.replay_wrapper)
                    sub_model = sub_model.model
                sub_models.append(sub_model)

        # place to save trained model
        log_path = "{}/{}/".format(args.log_folder, args.algo)
        save_path = os.path.join(log_path, "{}_{}".format(env_id, get_latest_run_id(log_path, env_id) + 1))
        params_path = "{}/{}".format(save_path, env_id)
        os.makedirs(params_path, exist_ok=True)

        # Train
        inner_model.sub_models = sub_models
        inner_model.replay_wrappers = replay_wrappers
        inner_model.args = args
        inner_model.save_folder = params_path
        if args.trained_agent_folder == '':
            model.learn(n_timesteps, **kwargs)
        else:
            model.eval(n_timesteps, **kwargs)

        # # Eval model for `n_eval_episodes` times
        # # env = gym.make(env_id)
        # # if env_wrapper is not None:
        # #     env = env_wrapper(env)
        # env = inner_model.env
        # ob = env.reset()
        # inner_model.macro_count = 0
        # inner_model.macro_act = None
        # episode_reward = 0.0
        # episode_rewards = []
        # ep_len = 0
        # macro_actions = []
        # macro_probs = []
        # actions = []
				# # For HER, monitor success rate
        # successes = []
        # # Record frames to create videos
        # need_record_frames = args.n_episodes_record_frames > 0
        # n_episodes_recorded = 0
        # rgb_arrays = []
        # rewards_each_step = []
        # if need_record_frames:
        #     from pyvirtualdisplay import Display
        #     display = Display(visible=0, size=(1400, 900))
        #     display.start()

        # while True:
        #     if need_record_frames:
        #         rgb = env.render('rgb_array')
        #         rgb = img_as_ubyte(resize(rgb, (rgb.shape[0]//2, rgb.shape[1]//2)))
        #         rgb_arrays.append(rgb)

        #     macro_action, action, _ = inner_model.predict(ob, deterministic=True)
        #     ob, reward, done, info = env.step(action)
        #     episode_reward += reward
        #     actions.append(action)
        #     ep_len += 1
        #     macro_actions.append(macro_action)
        #     rewards_each_step.append(reward)

        #     if done:
        #         if need_record_frames:
        #             current_save_folder = os.path.join(params_path, 'eval_ep' + str(n_episodes_recorded))
        #             print("Save frames to folder: %s" % current_save_folder)
        #             os.makedirs(current_save_folder, exist_ok=True)
        #             statistic_file = os.path.join(current_save_folder, 'statistic_file.txt')
        #             rgb_arrays_file = os.path.join(current_save_folder, 'rgb_arrays.pickle')
        #             actions_file = os.path.join(current_save_folder, 'actions.pickle')
        #             with open(statistic_file, 'w') as f:
        #                 ep_ret = sum(rewards_each_step)
        #                 f.write('%d: %f' % (n_episodes_recorded, ep_ret) + '\n')
        #                 d = {'macro_ac': macro_actions, 'rews_without_cost': rewards_each_step}
        #                 needed_keys = ['macro_ac', 'rews_without_cost']
        #                 for key in needed_keys:
        #                     f.write(key + '\n')
        #                     for v in d[key]:
        #                         f.write(str(v) + ' ')
        #                     f.write('\n\n')
        #             rgb_arrays = np.array(rgb_arrays)
        #             rgb_arrays.dump(rgb_arrays_file)
        #             actions = np.array(actions)
        #             print("shape of actions:", actions.shape)
        #             actions.dump(actions_file)
        #             n_episodes_recorded += 1

        #         if n_episodes_recorded >= args.n_episodes_record_frames:
        #             need_record_frames = False

        #         # NOTE: for env using VecNormalize, the mean reward
        #         # is a normalized reward when `--norm_reward` flag is passed
        #         episode_rewards.append(episode_reward)
        #         macro_probs.append(np.mean(macro_actions))
        #         episode_reward = 0.0
        #         ep_len = 0
        #         macro_actions = []
        #         rewards_each_step = []
        #         rgb_arrays = []
        #         actions = []
        #         inner_model.macro_count = 0
        #         inner_model.macro_act = None

        #         # For HER, record success rate
        #         maybe_is_success = info.get('is_success')
        #         if maybe_is_success is not None:
        #             successes.append(float(maybe_is_success))

        #         if len(episode_rewards) >= args.n_eval_episodes:
        #             break
        #         else:
        #             obs = env.reset()

        # print("=" * 70)
        # print("Evaluation result:")
        # print("Number of total eval episodes:", args.n_eval_episodes)
        # if len(successes) != 0:
        #     print("Number of success episodes:", np.sum(np.array(successes) == 1.0))
        # print("Mean episode reward:", np.mean(episode_rewards))
        # print("\% of macro using large policy:", macro_probs[:100])

        # # Save macro acts, episode return, success to file
        # eval_result_file = os.path.join(params_path, 'eval_result.txt')
        # with open(eval_result_file, 'w') as f:
        #     d = {'macro_ratio': macro_probs, 'return': episode_rewards, 'success': successes}
        #     for key in d:
        #         f.write(key + '\n')
        #         for v in d[key]:
        #             f.write(str(v) + ' ')
        #         f.write('\n\n')

        # Only save worker of rank 0 when using mpi
        if rank == 0:
            print("Saving to {}".format(save_path))

            model.save("{}/{}".format(save_path, env_id))
            for i, sub_model in enumerate(inner_model.sub_models):
                sub_model.save("{}/{}_sub{}".format(save_path, env_id, i))
            # Save hyperparams
            with open(os.path.join(params_path, 'config.yml'), 'a') as f:
                yaml.dump(saved_hyperparams, f)
                f.write('\n\n\n')
                yaml.dump(args, f)


            if normalize:
                # Unwrap
                if isinstance(env, VecFrameStack):
                    env = env.venv
                # Important: save the running average, for testing the agent we need that normalization
                env.save_running_average(params_path)
