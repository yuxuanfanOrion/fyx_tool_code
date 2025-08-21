import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
import cv2
import json

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    observations_to_image_vla,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info

# 添加数据保存相关的配置
VIDEO_DIR = '/hpc2hdd/home/yfan546/workplace/amap_ws/Falcon-main/data/video_orca3'
DATA_DIR = '/hpc2hdd/home/yfan546/workplace/amap_ws/Falcon-main/data/orca_data3'


class ORCAEvaluator(Evaluator):
    """
    Evaluator for Habitat environments with data recording capabilities.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加数据记录相关的实例变量
        self.episode_data = {}
        self.episode_step_count = defaultdict(int)
        # 添加成功状态记录
        self.episode_success_status = {}

    # -----------------------------
    # 数据处理和保存相关的辅助方法
    # -----------------------------
    def _process_depth_observation_for_video(self, obs_dict):
        """仅为视频生成处理深度观测数据，不影响原始数据保存"""
        processed_obs = obs_dict.copy()
        depth_key = "agent_0_articulated_agent_jaw_depth"
        if depth_key in processed_obs:
            depth = processed_obs[depth_key]
            if torch.is_tensor(depth):
                depth = depth.cpu().numpy()
            depth = np.clip(depth, 0, 10.0)
            depth_normalized = (depth / 10.0 * 255).astype(np.uint8)
            if len(depth_normalized.shape) == 2:
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            else:
                depth_colored = depth_normalized
            processed_obs[depth_key] = depth_colored
        return processed_obs

    def _get_action_mapping(self, action_value):
        """获取动作映射"""
        action_map = {
            0: "stop",
            1: "move_forward",
            2: "turn_left",
            3: "turn_right",
        }
        if isinstance(action_value, np.ndarray):
            action_value = action_value.item() if action_value.size == 1 else action_value[0]
        return action_map.get(action_value, f"unknown_{action_value}")

    def _get_episode_dir_name(self, scene_id, episode_id, eval_count, success_status=None):
        """生成episode目录名，包含成功状态"""
        if isinstance(scene_id, str):
            scene_name = scene_id.split('/')[-1].split('.')[0]
        else:
            scene_name = str(scene_id)
        
        if success_status is not None:
            success_suffix = f"_success={success_status:.1f}"
        else:
            success_suffix = ""
        
        return f"{scene_name}_episode_{episode_id}_{eval_count}{success_suffix}"

    def _rename_episode_directory(self, old_episode_key, success_status):
        """根据成功状态重命名episode目录"""
        scene_id, episode_id, eval_count = old_episode_key
        
        # 旧目录名（没有success状态）
        old_dir_name = self._get_episode_dir_name(scene_id, episode_id, eval_count)
        old_dir_path = os.path.join(DATA_DIR, old_dir_name)
        
        # 新目录名（包含success状态）
        new_dir_name = self._get_episode_dir_name(scene_id, episode_id, eval_count, success_status)
        new_dir_path = os.path.join(DATA_DIR, new_dir_name)
        
        # 如果旧目录存在，重命名为新目录
        if os.path.exists(old_dir_path) and old_dir_path != new_dir_path:
            try:
                os.rename(old_dir_path, new_dir_path)
                print(f"Renamed episode directory: {old_dir_name} -> {new_dir_name}")
            except OSError as e:
                print(f"Failed to rename directory {old_dir_path} to {new_dir_path}: {e}")

    def _save_raw_images_before_transforms(self, batch, current_episodes_info, ep_eval_count, step_data, step_counter):
        """在apply_obs_transforms_batch之前保存原始图像"""
        for i in range(len(current_episodes_info)):
            episode_key = self._get_episode_key(current_episodes_info[i], ep_eval_count)
            
            scene_id, episode_id, eval_count = episode_key
            episode_dir_name = self._get_episode_dir_name(scene_id, episode_id, eval_count)
            episode_dir = os.path.join(DATA_DIR, episode_dir_name)
            os.makedirs(episode_dir, exist_ok=True)

            # 获取动作名称
            if step_data is not None and i < len(step_data):
                action_value = step_data[i]
            else:
                action_value = 0  # 初始步骤
            action_name = self._get_action_mapping(action_value)
            
            # 使用全局步骤计数器
            step_str = f"{step_counter:03d}"
            
            # 获取当前环境的观测数据
            env_obs = {k: v[i] for k, v in batch.items()}

            # 保存RGB图像
            rgb_key = "agent_0_articulated_agent_jaw_rgb"
            if rgb_key in env_obs:
                rgb_image = env_obs[rgb_key]
                if torch.is_tensor(rgb_image):
                    rgb_image = rgb_image.cpu().numpy()
                
                # 确保数据格式正确
                if rgb_image.dtype != np.uint8:
                    if rgb_image.max() <= 1.0:  # 如果是0-1范围
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    else:
                        rgb_image = rgb_image.astype(np.uint8)
                
                rgb_filename = f"{step_str}_{action_name}_FrontView.png"
                rgb_path = os.path.join(episode_dir, rgb_filename)
                cv2.imwrite(rgb_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

            # 保存深度图像（原始深度值转换为可视化PNG）
            depth_key = "agent_0_articulated_agent_jaw_depth"
            if depth_key in env_obs:
                depth_image = env_obs[depth_key]
                if torch.is_tensor(depth_image):
                    depth_image = depth_image.cpu().numpy()
                
                # 保存原始深度数据为numpy文件
                depth_raw_filename = f"{step_str}_{action_name}_FrontView_depth_raw.npy"
                depth_raw_path = os.path.join(episode_dir, depth_raw_filename)
                np.save(depth_raw_path, depth_image)
                
                # 同时保存深度的PNG可视化版本
                # 将深度值转换为可视化的图像
                depth_vis = depth_image.copy()
                # 处理无效值
                depth_vis = np.nan_to_num(depth_vis, nan=0.0, posinf=10.0, neginf=0.0)
                # 裁剪到合理范围
                depth_vis = np.clip(depth_vis, 0, 10.0)
                # 归一化到0-255
                if depth_vis.max() > 0:
                    depth_vis = (depth_vis / depth_vis.max() * 255).astype(np.uint8)
                else:
                    depth_vis = depth_vis.astype(np.uint8)
                
                depth_filename = f"{step_str}_{action_name}_FrontView_depth.png"
                depth_path = os.path.join(episode_dir, depth_filename)
                cv2.imwrite(depth_path, depth_vis)

    def _save_images(self, episode_key, step_idx, batch, env_idx, action_value):
        """保存RGB和深度图像，使用原始未处理的观测数据"""
        scene_id, episode_id, eval_count = episode_key
        episode_dir_name = self._get_episode_dir_name(scene_id, episode_id, eval_count)
        episode_dir = os.path.join(DATA_DIR, episode_dir_name)
        os.makedirs(episode_dir, exist_ok=True)

        action_name = self._get_action_mapping(action_value)
        step_str = f"{step_idx:03d}"
        env_obs = {k: v[env_idx] for k, v in batch.items()}

        # 保存RGB图像
        rgb_key = "agent_0_articulated_agent_jaw_rgb"
        if rgb_key in env_obs:
            rgb_image = env_obs[rgb_key]
            if torch.is_tensor(rgb_image):
                rgb_image = rgb_image.cpu().numpy()
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            rgb_filename = f"{step_str}_{action_name}_FrontView.png"
            rgb_path = os.path.join(episode_dir, rgb_filename)
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        # 保存深度图像 - 完全不做任何处理，直接保存原始depth
        depth_key = "agent_0_articulated_agent_jaw_depth"
        if depth_key in env_obs:
            depth_image = env_obs[depth_key]
            if torch.is_tensor(depth_image):
                depth_image = depth_image.cpu().numpy()
            
            # 不做任何处理，直接保存原始深度数据
            depth_filename = f"{step_str}_{action_name}_FrontView_depth.npy"
            depth_path = os.path.join(episode_dir, depth_filename)
            np.save(depth_path, depth_image)

    def _get_point_goal_info(self, batch, env_idx):
        """获取点目标信息"""
        point_goal_data = {"distance": 0.0, "angle": 0.0}
        pointgoal_key = 'agent_0_pointgoal_with_gps_compass'
        if pointgoal_key in batch:
            pointgoal_obs = batch[pointgoal_key]
            if torch.is_tensor(pointgoal_obs):
                pointgoal_obs = pointgoal_obs.cpu().numpy()
            if env_idx < len(pointgoal_obs):
                env_pointgoal = pointgoal_obs[env_idx]
                if len(env_pointgoal) >= 2:
                    point_goal_data["distance"] = float(env_pointgoal[0])
                    point_goal_data["angle"] = float(env_pointgoal[1])
        return point_goal_data

    def _save_episode_data(self, episode_key, success_status=None):
        """保存episode的完整数据"""
        scene_id, episode_id, eval_count = episode_key
        episode_dir_name = self._get_episode_dir_name(scene_id, episode_id, eval_count, success_status)
        episode_dir = os.path.join(DATA_DIR, episode_dir_name)
        if episode_key not in self.episode_data:
            return
        episode_info = self.episode_data[episode_key]
        os.makedirs(episode_dir, exist_ok=True)

        # 保存动作序列
        with open(os.path.join(episode_dir, "actions.json"), "w") as f:
            json.dump(episode_info["actions"], f, indent=2)

        # 保存点目标信息
        with open(os.path.join(episode_dir, "point_goal.json"), "w") as f:
            json.dump(episode_info["point_goals"], f, indent=2)

        # 保存指令文本
        instruction_text = (
            "You are a navigation robot with social navigation capabilities. "
            "You need to navigate to the corresponding location based on the given "
            "images and target point position, while trying to avoid moving people. "
            "Each step's instruction needs to combine the current images and the "
            "point_goal position."
        )
        with open(os.path.join(episode_dir, "instruction.txt"), "w") as f:
            f.write(instruction_text)
        with open(os.path.join(episode_dir, "instruction.json"), "w") as f:
            json.dump({"instruction": instruction_text, "steps": len(episode_info["actions"])}, f, indent=2)

    def _get_episode_key(self, episode_info, ep_eval_count):
        """生成episode的唯一标识"""
        return (
            episode_info.scene_id,
            episode_info.episode_id,
            ep_eval_count[(episode_info.scene_id, episode_info.episode_id)],
        )

    def _initialize_episode_data(self, episode_info, ep_eval_count, batch, env_idx):
        """初始化episode数据收集"""
        episode_key = self._get_episode_key(episode_info, ep_eval_count)
        if episode_key not in self.episode_data:
            self.episode_data[episode_key] = {"actions": [], "point_goals": []}
            self.episode_step_count[episode_key] = 0
            # 保存初始帧
            self._save_images(episode_key, self.episode_step_count[episode_key], batch, env_idx, 0)
            # 保存初始点目标信息
            point_goal_info = self._get_point_goal_info(batch, env_idx)
            self.episode_data[episode_key]["point_goals"].append(point_goal_info)
        return episode_key

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        success_cal = 0 ## my added
        # 添加全局步骤计数器
        global_step_counter = 0
        
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        
        # 在apply_obs_transforms_batch之前保存原始图像
        current_episodes_info = envs.current_episodes()
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)
        self._save_raw_images_before_transforms(batch, current_episodes_info, ep_eval_count, None, global_step_counter)
        global_step_counter += 1
        
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        # 初始化视频帧记录（支持VLA格式）
        if len(config.habitat_baselines.eval.video_option) > 0:
            rgb_frames: List[List[np.ndarray]] = []
            vla_frames: List[List[np.ndarray]] = []
            for env_idx in range(config.habitat_baselines.num_environments):
                env_obs = {k: v[env_idx] for k, v in batch.items()}
                # 只在生成视频时才处理深度数据
                processed_obs = self._process_depth_observation_for_video(env_obs)
                frame = observations_to_image(processed_obs, {})
                frame_vla = observations_to_image_vla(processed_obs, {})
                rgb_frames.append([frame])
                vla_frames.append([frame_vla])
        else:
            rgb_frames = None
            vla_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        # 添加动作记录
        actions_record = defaultdict(list)
        agent.eval()

        # 创建数据保存目录
        os.makedirs(DATA_DIR, exist_ok=True)

        # 初始化当前episode的数据收集
        for i in range(envs.num_envs):
            self._initialize_episode_data(current_episodes_info[i], ep_eval_count, batch, i)

        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with inference_mode():
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    **space_lengths,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if hasattr(agent, '_agents') and agent._agents[0]._actor_critic.action_distribution_type == 'categorical':
                step_data = [a.numpy() for a in action_data.env_actions.cpu()]
            elif is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            # 记录动作和步数统计
            for i in range(envs.num_envs):
                episode_key = self._get_episode_key(current_episodes_info[i], ep_eval_count)
                action_value = step_data[i]
                if isinstance(action_value, np.ndarray):
                    stored_action = action_value.item() if action_value.size == 1 else int(action_value[0])
                else:
                    stored_action = int(action_value)

                # 记录动作到actions_record
                actions_record[episode_key].append({"type": "array", "value": np.array(action_value).tolist()})

                # 记录到episode_data
                if episode_key in self.episode_data:
                    self.episode_data[episode_key]["actions"].append(stored_action)
                    self.episode_step_count[episode_key] += 1

            outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            
            # 在apply_obs_transforms_batch之前保存原始图像
            self._save_raw_images_before_transforms(batch, current_episodes_info, ep_eval_count, step_data, global_step_counter)
            global_step_counter += 1
            
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs

            # 保存step后的点目标信息
            for i in range(n_envs):
                episode_key = self._get_episode_key(current_episodes_info[i], ep_eval_count)
                if episode_key in self.episode_data:
                    point_goal_info = self._get_point_goal_info(batch, i)
                    self.episode_data[episode_key]["point_goals"].append(point_goal_info)

            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if len(config.habitat_baselines.eval.video_option) > 0:
                    # 处理观测数据并生成视频帧 - 只在这里处理深度用于视频
                    env_obs = {k: v[i] for k, v in batch.items()}
                    processed_obs = self._process_depth_observation_for_video(env_obs)
                    frame = observations_to_image(processed_obs, disp_info)
                    vla_frame = observations_to_image_vla(processed_obs, disp_info)

                    if not not_done_masks[i].any().item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        final_obs = {k: v[i] * 0.0 for k, v in batch.items()}
                        final_processed = self._process_depth_observation_for_video(final_obs)
                        final_frame = observations_to_image(final_processed, disp_info)
                        final_frame = overlay_frame(final_frame, disp_info)
                        final_vla_frame = observations_to_image_vla(final_processed, disp_info)
                        final_vla_frame = overlay_frame(final_vla_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        rgb_frames[i].append(frame)
                        vla_frames[i].append(final_vla_frame)
                        vla_frames[i].append(vla_frame)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        vla_frame = overlay_frame(vla_frame, disp_info)
                        rgb_frames[i].append(frame)
                        vla_frames[i].append(vla_frame)

                # episode ended
                if not not_done_masks[i].any().item():
                    pbar.update()
                    # 获取成功状态
                    success_status = 1.0 if disp_info.get('success', False) else 0.0
                    
                    if "success" in disp_info:
                        success_cal += disp_info['success']
                        print(f"Till now Success Rate: {success_cal/(len(stats_episodes)+1)}")
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )

                    # 保存episode数据并重命名目录以包含成功状态
                    episode_key = self._get_episode_key(current_episodes_info[i], ep_eval_count)
                    # 先保存数据到临时目录，然后重命名包含成功状态
                    self._save_episode_data(episode_key)
                    self._rename_episode_directory(episode_key, success_status)

                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    # 清理episode缓存
                    if episode_key in self.episode_data:
                        del self.episode_data[episode_key]
                    if episode_key in self.episode_step_count:
                        del self.episode_step_count[episode_key]

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        scene_id = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        print(f"This is Scene ID: {scene_id}, Episode ID: {current_episodes_info[i].episode_id}.")

                        # 生成标准视频
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=VIDEO_DIR,
                            # Since the final frame is the start frame of the next episode.
                            images=rgb_frames[i][:-1],
                            scene_id=f"{current_episodes_info[i].scene_id}".split('/')[-1].split('.')[0],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # 生成VLA格式视频
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            images=vla_frames[i][:-1],
                            scene_id="vla_data_" + f"{current_episodes_info[i].scene_id}".split('/')[-1].split('.')[0],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # Since the starting frame of the next episode is the final frame.
                        rgb_frames[i] = rgb_frames[i][-1:]
                        vla_frames[i] = vla_frames[i][-1:]

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
                vla_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
                vla_frames,
            )

            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

            # 为新开始的episode初始化数据收集
            new_episodes_info = envs.current_episodes()
            for i in range(envs.num_envs):
                episode_key = self._get_episode_key(new_episodes_info[i], ep_eval_count)
                if episode_key not in self.episode_data:
                    self._initialize_episode_data(new_episodes_info[i], ep_eval_count, batch, i)

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        # ==== 保存 result.json ====
        result_path = os.path.join("output/", "result.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        evalai_result = {
            "SR": round(aggregated_stats.get("success", 0), 4),
            "SPL": round(aggregated_stats.get("spl", 0), 4),
            "PSC": round(aggregated_stats.get("psc", 0), 4),
            "H-Coll": round(aggregated_stats.get("human_collision", 0), 4),
            "Total": round(
                0.4 * aggregated_stats.get("success", 0)
                + 0.3 * aggregated_stats.get("spl", 0)
                + 0.3 * aggregated_stats.get("psc", 0),
                4,
            ),
        }
        with open(result_path, "w") as f:
            json.dump(evalai_result, f, indent=2)

        # ==== 保存 actions.json ====
        actions_output_path = os.path.join("output/", "actions.json")
        os.makedirs(os.path.dirname(actions_output_path), exist_ok=True)
        serializable_actions = {
            f"{scene_id}|{episode_id}|{eval_count}": actions
            for (scene_id, episode_id, eval_count), actions in actions_record.items()
        }
        with open(actions_output_path, "w") as f:
            json.dump(serializable_actions, f, indent=2)
