"""The jumper hexapod family.

`common/` holds everything the jumper tasks share (robot definition, environment
skeleton, MDP terms, PPO baseline); every other subdirectory is one task:

    tasks/jumper/flat/       task id  jumper.flat       no gait prior, the control group
    tasks/jumper/tripod/     task id  jumper.tripod     tripod gait
    tasks/jumper/tetrapod/   task id  jumper.tetrapod   tetrapod gait
    tasks/jumper/ripple/     task id  jumper.ripple     ripple gait
    tasks/jumper/jump/       task id  jumper.jump       reference-guided high jump (high_jump_flat.npz)
    tasks/jumper/ref_free_jump/  task id  jumper.ref_free_jump  the same jump, no recording on the robot

The robot is a 22-DoF heterogeneous hexapod: two 5-DoF front arms (in FOOT mode
the grippers are locked and it walks on fixed jaws) plus four 3-DoF middle and rear
legs.
"""
