import sys
import numpy as np
import cv2
import gymnasium as gym
import time

from tetris_gymnasium.envs import Tetris


if __name__ == "__main__":
    # Create an instance of Tetris
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset()

    terminated = False
    while not terminated:
        env.render()
        action = None
        while action is None:
            key = cv2.waitKey(1)
            time.sleep(0.1)  # Add a small delay to make the game playable

            # if key == ord("a"):
            #     action = env.unwrapped.actions.move_left
            # elif key == ord("d"):
            #     action = env.unwrapped.actions.move_right
            # elif key == ord("s"):
            #     action = env.unwrapped.actions.move_down
            # elif key == ord("w"):
            #     action = env.unwrapped.actions.rotate_counterclockwise
            # elif key == ord("e"):
            #     action = env.unwrapped.actions.rotate_clockwise
            # elif key == ord(" "):
            #     action = env.unwrapped.actions.hard_drop
            # elif key == ord("q"):
            #     action = env.unwrapped.actions.swap
            # elif key == ord("r"):
            #     action = env.unwrapped.actions.no_op
            #     break

            # if (
            #     cv2.getWindowProperty(env.unwrapped.window_name, cv2.WND_PROP_VISIBLE)
            #     == 0
            # ):
            #     sys.exit()
            action = env.action_space.sample()  # Random action for testinga
        obs, reward, terminated, truncated, info = env.step(action)
