"""MDP terms specific to jumper.five_foot.

`events` holds what puts and keeps the left-front arm in its carried pose;
`gripper` is the only thing that moves the claw afterwards, and only in replay;
`rewards` holds the terms that exist because the robot is walking on five legs;
`pose_command` is the attitude command those terms measure against and the
gravity convention they share; `curriculum` decides when the claw starts carrying
something. Everything the family shares -- gating, phase matching, the mirror
transform, the command curriculum -- stays in `tasks/jumper/common/mdp/`.
"""

from . import curriculum, events, gripper, pose_command, rewards

__all__ = ["curriculum", "events", "gripper", "pose_command", "rewards"]
