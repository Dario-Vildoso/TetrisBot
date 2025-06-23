import sys
import numpy as np
import cv2
import gymnasium as gym

from tetris_gymnasium.envs import Tetris

if __name__ == "__main__":
    # Create an instance of Tetris
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.unwrapped.rewards.alife = 0
    env.unwrapped.rewards.clear_line = 10
    env.unwrapped.rewards.game_over = -2
    env.unwrapped.rewards.invalid_action = -0.1
    board = env.unwrapped.board.copy()
    playable_board = board[:-4, 4:-4]
    # Set the bottom row to 2s
    playable_board[-1, :] = 2
    # Create a gap at column 5 (index 4)
    playable_board[-1, 4] = 0
    # Put the modified board back (with padding)
    board[:-4, 4:-4] = playable_board
    obs, _ = env.reset()
    env.unwrapped.board = board.copy()

    # Fill the bottom row with 2s except for one gap (e.g., column 5)
    board = obs["board"]
    # Main game loop
    terminated = False
    while not terminated:

        # Render the current state of the game as text
        env.render()

        # Pick an action from user input mapped to the keyboard
        action = None
        while action is None:
            key = cv2.waitKey(1)

            if key == ord("a"):
                action = env.unwrapped.actions.move_left
            elif key == ord("d"):
                action = env.unwrapped.actions.move_right
            elif key == ord("s"):
                action = env.unwrapped.actions.move_down
            elif key == ord("w"):
                action = env.unwrapped.actions.rotate_counterclockwise
            elif key == ord("e"):
                action = env.unwrapped.actions.rotate_clockwise
            elif key == ord(" "):
                action = env.unwrapped.actions.hard_drop
            elif key == ord("q"):
                action = env.unwrapped.actions.swap
            elif key == ord("r"):
                action = env.unwrapped.actions.no_op
                break

            if (
                cv2.getWindowProperty(env.unwrapped.window_name, cv2.WND_PROP_VISIBLE)
                == 0
            ):
                sys.exit()

        # Perform the action
        obs, reward, terminated, truncated, info = env.step(action)
        board = obs["board"]
        board = board.copy()
        # board[board > 1] = 2
        board = board[:-4, 4:-4]
        piece = obs["active_tetromino_mask"]
        piece = piece[:-4, 4:-4]
        mask = (board > 1) & (piece == 1)
        piece_id = np.max(board[mask == 1])
        piece = np.where(mask, 1, 0)
        piece_coords = np.argwhere(piece == 1)
        board[piece == 1] = 0
        print(
            f"Observation: {board}\n"
            f"pieza: {piece}\n"
            f"pieza id: {piece_id}\n"
            f"pieza coords: {piece_coords}\n"
            f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
        )

    # Game over
    print("Game Over!")