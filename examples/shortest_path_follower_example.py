#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import cv2
import numpy as np

import habitat
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

cv2 = try_cv2_import()


import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=os.path.join("examples", "images"),
        required=True,
        help="output directory to store recorded data ",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to collect data for",
    )
    args = parser.parse_args()
    return args


args = get_args()

IMAGE_DIR = args.out_dir
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def shortest_path_example():
    config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.freeze()

    with SimpleRLEnv(config=config) as env:
        goal_radius = env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
        follower = ShortestPathFollower(
            env.habitat_env.sim, goal_radius, False
        )
        # print(env.scene_name)
        print("Environment creation successful")
        for episode in range(args.num_episodes):
            env.reset()
            dirname = os.path.join(
                IMAGE_DIR, "shortest_path_example", "%02d" % episode
            )
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)
            os.makedirs(os.path.join(dirname, "images"))
            os.makedirs(os.path.join(dirname, "topdownmap"))
            print("Agent stepping around inside environment.")
            # print(env.habitat_env.scene_id)
            images, actions = [], []
            step = 0
            while not env.habitat_env.episode_over:
                best_action = follower.get_next_action(
                    env.habitat_env.current_episode.goals[0].position
                )
                if best_action is None:
                    break

                observations, reward, done, info = env.step(best_action)
                im = observations["rgb"]
                top_down_map = draw_top_down_map(info, im.shape[0])
                output_im = np.concatenate((im, top_down_map), axis=1)
                images.append(output_im)

                # write image at step k 
                cv2.imwrite(os.path.join(dirname, "images", "%03d.png"%step), im)
                # write top-down map separately at step k 
                cv2.imwrite(os.path.join(dirname, "topdownmap", "%03d.png"%step), top_down_map)
                actions.append(best_action)
                step += 1
            np.save(os.path.join(dirname, "shortest_path_actions_taken.npy"), np.array(actions))
            images_to_video(images, dirname, "trajectory")
            print("Episode finished")


def main():
    shortest_path_example()


if __name__ == "__main__":
    main()
