import copy
import gc
import json
import os
import random
import re
import sys
import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union

import cv2
import lmdb
import msgpack_numpy
import numpy as np
import torch
import tqdm
from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    get_action_space_info,
    generate_video,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from habitat.utils.visualizations.utils import (
    observations_to_image,
    observations_to_image_vla,
    overlay_frame,
)
from PIL import Image

# LLaVA相关导入
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


def extract_instruction_tokens(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    tokens_uuid: str = "tokens",
) -> Dict[str, Any]:
    """
    Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure.
    """
    if instruction_sensor_uuid not in observations[0] or instruction_sensor_uuid == "pointgoal_with_gps_compass":
        return observations
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and tokens_uuid in observations[i][instruction_sensor_uuid]
        ):
            observations[i][instruction_sensor_uuid] = observations[i][instruction_sensor_uuid]["tokens"]
        else:
            break
    return observations


def sample_and_pad_images(images, num_frames=8, width=512, height=512):
    """从NaVILATrainer中复制的图像采样和填充函数"""
    frames = copy.deepcopy(images)

    if len(frames) < num_frames:
        padding_frames = num_frames - len(frames)
        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
    else:
        padding_frames = 0

    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]

    return sampled_frames


class NaVILAEvaluator(Evaluator):
    """
    NaVILA模型的评估器，继承自Evaluator并对标FALCONEvaluator的完整流程
    """

    def __init__(self, config=None, num_chunks=1, chunk_idx=0):
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        self.config = config

    def _process_observation(self, obs_dict):
        """处理观察数据，特别是处理RGB和深度图像"""
        processed_obs = obs_dict.copy()
        
        # 处理RGB观察
        rgb_key = "agent_0_articulated_agent_jaw_rgb"
        if rgb_key in processed_obs:
            rgb = processed_obs[rgb_key]
            if torch.is_tensor(rgb):
                rgb = rgb.cpu().numpy()
            processed_obs[rgb_key] = rgb
            
        # 处理深度观察
        depth_key = "agent_0_articulated_agent_jaw_depth"
        if depth_key in processed_obs:
            depth = processed_obs[depth_key]
            
            if torch.is_tensor(depth):
                depth = depth.cpu().numpy()
            
            # 确保深度值在合理范围内
            depth = np.clip(depth, 0, 10.0)
            
            # 归一化到0-255范围
            depth_normalized = (depth / 10.0 * 255).astype(np.uint8)
            
            # 如果是单通道，转换为3通道用于可视化
            if len(depth_normalized.shape) == 2:
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            else:
                depth_colored = depth_normalized
            
            processed_obs[depth_key] = depth_colored
            
        return processed_obs

    def _load_navila_model(self, checkpoint_path: str):
        """加载NaVILA模型"""
        model_name = os.path.basename(os.path.normpath(checkpoint_path))
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            checkpoint_path, model_name
        )
        model = model.cuda()
        return tokenizer, model, image_processor, context_len

    def _generate_navila_action(self, model, tokenizer, image_processor, 
                              past_and_current_rgbs, instruction):
        """使用NaVILA模型生成动作"""
        with torch.no_grad():
            num_video_frames = model.config.num_video_frames
            past_and_current_rgbs = sample_and_pad_images(
                past_and_current_rgbs, num_frames=num_video_frames
            )

            interleaved_images = "<image>\n" * (len(past_and_current_rgbs) - 1)

            question = (
                f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
                f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: "{instruction}" '
                f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
                f"degree, moving forward a certain distance, or stop if the task is completed."
            )

            conv_mode = "llama_3"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            images_tensor = process_images(
                past_and_current_rgbs, image_processor, model.config
            ).to(model.device, dtype=torch.float16)
            
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor.half().cuda(),
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=32,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=tokenizer.eos_token_id,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()

            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            return outputs

    def _parse_action_from_output(self, output_text):
        """从模型输出解析动作"""
        patterns = {
            0: re.compile(r"\bstop\b", re.IGNORECASE),
            1: re.compile(r"\bmove forward\b", re.IGNORECASE),
            2: re.compile(r"\bturn left\b", re.IGNORECASE),
            3: re.compile(r"\bturn right\b", re.IGNORECASE),
        }

        for action, pattern in patterns.items():
            if pattern.search(output_text):
                return action
            
        logger.warn(f"Could not parse action from model output: '{output_text}'. Defaulting to action 1 (move forward).")
        return 1

    def _process_action_with_distance_and_angle(self, action, output_text):
        """处理动作的距离和角度参数"""
        queue_actions = []
        
        if action == 1:  # 前进
            try:
                match = re.search(r"move forward (\d+) cm", output_text)
                distance = int(match.group(1))
            except:
                distance = 25
            if (distance % 25) != 0:
                distance = min([25, 50, 75], key=lambda x: abs(x - distance))
            
            for _ in range(int(distance // 25) - 1):
                queue_actions.append(1)
                
        elif action == 2:  # 左转
            try:
                match = re.search(r"turn left (\d+) degree", output_text)
                degree = int(match.group(1))
            except:
                degree = 15
            if (degree % 15) != 0:
                degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                
            for _ in range(int(degree // 15) - 1):
                queue_actions.append(2)
                
        elif action == 3:  # 右转
            try:
                match = re.search(r"turn right (\d+) degree", output_text)
                degree = int(match.group(1))
            except:
                degree = 15
            if (degree % 15) != 0:
                degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                
            for _ in range(int(degree // 15) - 1):
                queue_actions.append(3)

        return queue_actions

    def _get_action_instruction(self, batch, current_episodes_info):
        """根据传感器数据为LLM生成合适的指令"""
        pointgoal_sensor_key = 'agent_0_pointgoal_with_gps_compass'

        # 优先尝试获取基于文本的指令 (适用于 VLN 等任务)
        try:
            instruction = current_episodes_info[0].instruction.instruction_text
            if instruction:
                return instruction
        except (AttributeError, IndexError):
            pass

        # 如果没有文本指令，则尝试从 PointGoal 传感器生成指令
        if pointgoal_sensor_key in batch:
            pointgoal_obs = batch[pointgoal_sensor_key][0].cpu().numpy()
            distance = pointgoal_obs[0]
            angle_rad = pointgoal_obs[1]
            return f"Your goal is at a distance of {distance:.2f} meters and at a relative angle of {angle_rad:.2f} radians."

        # 如果两种指令源都不可用，返回一个通用的默认指令
        return "Navigate to the designated goal."

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
        """评估NaVILA代理 - 完整对标FALCONEvaluator"""
        
        # 加载NaVILA模型
        checkpoint_path = '/hpc2hdd/home/yfan546/workplace/amap_ws/Falcon-main/pretrained_model/navila-llama3-8b-8f'
        tokenizer, model, image_processor, context_len = self._load_navila_model(checkpoint_path)
        
        success_cal = 0
        observations = envs.reset()
        observations = envs.post_step(observations)
        
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)

        # 获取动作空间信息 - 对标FALCONEvaluator
        action_shape, discrete_actions = get_action_space_info(
            env_spec.action_space
        )

        # 初始化状态变量 - 对标FALCONEvaluator
        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")
        
        not_done_masks = torch.zeros(
            envs.num_envs,
            1,  # masks_shape
            device=device,
            dtype=torch.bool,
        )

        # 初始化NaVILA特有的状态
        past_rgbs = [[] for _ in range(envs.num_envs)]
        queue_actions = [[] for _ in range(envs.num_envs)]  # 每个环境独立的动作队列
        
        stats_episodes: Dict[Any, Any] = {}
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        # 初始化视频帧 - 对标FALCONEvaluator
        if len(config.habitat_baselines.eval.video_option) > 0:
            rgb_frames: List[List[np.ndarray]] = []
            for env_idx in range(envs.num_envs):
                env_obs = {k: v[env_idx] for k, v in batch.items()}
                processed_obs = self._process_observation(env_obs)
                frame = observations_to_image(processed_obs, {})
                rgb_frames.append([frame])
        else:
            rgb_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        # 设置评估episode数量 - 对标FALCONEvaluator
        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = getattr(config.habitat_baselines.eval, 'evals_per_ep', 1)
        
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
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
        actions_record = defaultdict(list)
        rgb_key = "agent_0_articulated_agent_jaw_rgb"
        
        # 主评估循环 - 对标FALCONEvaluator
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()

            # 生成动作 - NaVILA特有逻辑
            step_data = []
            for env_idx in range(envs.num_envs):
                if len(queue_actions[env_idx]) > 0:
                    # 使用队列中的动作
                    action = queue_actions[env_idx].pop(0)
                else:
                    # 使用NaVILA模型生成动作
                    with torch.no_grad():
                        curr_rgb_numpy = batch[rgb_key][env_idx].cpu().numpy().copy()
                        curr_rgb = Image.fromarray(np.uint8(curr_rgb_numpy)).convert("RGB")
                        past_and_current_rgbs = past_rgbs[env_idx] + [curr_rgb]

                        instruction = self._get_action_instruction(batch, current_episodes_info)
                        
                        # 生成动作
                        output_text = self._generate_navila_action(
                            model, tokenizer, image_processor, 
                            past_and_current_rgbs, instruction
                        )
                        
                        action = self._parse_action_from_output(output_text)
                        queue_actions[env_idx] = self._process_action_with_distance_and_angle(action, output_text)
                # [3,]
                step_data.append(action)
            

            step_data = [np.array(step_data)]
            
            outputs = envs.step(step_data)
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]

            # 记录动作 - 对标FALCONEvaluator
            for i in range(envs.num_envs):
                episode_key = (
                    current_episodes_info[i].scene_id,
                    current_episodes_info[i].episode_id,
                    ep_eval_count[
                        (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    ]
                )

                action_value = step_data[i]
                if isinstance(action_value, np.ndarray):
                    stored_action = {
                        "type": "array",
                        "value": action_value.tolist()
                    }
                else:
                    stored_action = {
                        "type": "array", 
                        "value": np.array(action_value).tolist()
                    }

                actions_record[episode_key].append(stored_action)

            # 后处理观察 - 对标FALCONEvaluator
            observations = extract_instruction_tokens(
                observations,
                getattr(config.habitat.task, 'instruction_sensor_uuid', "instruction")
            )
            observations = envs.post_step(observations)
            batch = batch_obs(observations, device=device)
            batch = apply_obs_transforms_batch(batch, obs_transforms)

            # 更新masks和rewards - 对标FALCONEvaluator
            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, 1)  # masks_shape

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards

            # 获取下一个episode信息
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs

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

                # 排除rank0_keys中的信息用于显示
                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                # 视频帧处理 - 对标FALCONEvaluator
                if len(config.habitat_baselines.eval.video_option) > 0:
                    env_obs = {k: v[i] for k, v in batch.items()}
                    processed_obs = self._process_observation(env_obs)
                    
                    frame = observations_to_image(processed_obs, disp_info)
                    
                    if not not_done_masks[i].any().item():
                        # episode结束时的处理
                        final_obs = {k: v[i] * 0.0 for k, v in batch.items()}
                        final_processed = self._process_observation(final_obs)
                        final_frame = observations_to_image(final_processed, disp_info)
                        final_frame = overlay_frame(final_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        rgb_frames[i].append(frame)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        rgb_frames[i].append(frame)

                # 更新历史RGB
                if rgb_key in batch:
                    rgb_numpy = batch[rgb_key][i].cpu().numpy().copy()
                    past_rgbs[i].append(Image.fromarray(rgb_numpy).convert("RGB"))

                # episode结束处理 - 对标FALCONEvaluator
                if not not_done_masks[i].any().item():
                    pbar.update()
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
                    ep_eval_count[k] += 1
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    # 重置NaVILA特有状态
                    past_rgbs[i] = []
                    queue_actions[i] = []

                    # 生成视频 - 对标FALCONEvaluator
                    if len(config.habitat_baselines.eval.video_option) > 0:
                        scene_id = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        print(f"This is Scene ID: {scene_id}, Episode ID: {current_episodes_info[i].episode_id}.")
                        
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            images=rgb_frames[i][:-1],
                            scene_id=f"{current_episodes_info[i].scene_id}".split('/')[-1].split('.')[0],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # 重置帧序列
                        rgb_frames[i] = rgb_frames[i][-1:]

                    # GFX重放数据保存 - 对标FALCONEvaluator
                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            # 环境暂停处理 - 对标FALCONEvaluator
            not_done_masks = not_done_masks.to(device=device)
            
            # 自定义pause_envs函数，适配NaVILA的特殊状态
            if len(envs_to_pause) > 0:
                (
                    envs,
                    not_done_masks,
                    current_episode_reward,
                    batch,
                    rgb_frames,
                ) = self._pause_envs_navila(
                    envs_to_pause,
                    envs,
                    not_done_masks,
                    current_episode_reward,
                    batch,
                    rgb_frames,
                    past_rgbs,
                    queue_actions,
                )

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        # 聚合统计 - 对标FALCONEvaluator
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

        # 写入tensorboard - 对标FALCONEvaluator
        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        # 保存结果 - 对标FALCONEvaluator
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

        # 保存动作记录 - 对标FALCONEvaluator
        actions_output_path = os.path.join("output/", "actions.json")
        os.makedirs(os.path.dirname(actions_output_path), exist_ok=True)
        serializable_actions = {
            f"{scene_id}|{episode_id}|{eval_count}": actions
            for (scene_id, episode_id, eval_count), actions in actions_record.items()
        }
        with open(actions_output_path, "w") as f:
            json.dump(serializable_actions, f, indent=2)

    def _pause_envs_navila(
        self,
        envs_to_pause,
        envs,
        not_done_masks,
        current_episode_reward,
        batch,
        rgb_frames,
        past_rgbs,
        queue_actions,
    ):
        """自定义的环境暂停函数，处理NaVILA特有的状态"""
        
        # 使用原有的pause_envs函数处理基础状态
        (
            envs,
            not_done_masks,
            current_episode_reward,
            batch,
            rgb_frames,
        ) = pause_envs(
            envs_to_pause,
            envs,
            not_done_masks,
            current_episode_reward,
            None,  # prev_actions (NaVILA不需要)
            batch,
            rgb_frames,
            None,  # vla_frames (NaVILA不需要)
        )
        
        # 处理NaVILA特有的状态
        if past_rgbs is not None:
            for env_idx in sorted(envs_to_pause, reverse=True):
                del past_rgbs[env_idx]
                del queue_actions[env_idx]
        
        return (
            envs,
            not_done_masks,
            current_episode_reward,
            batch,
            rgb_frames,
        )