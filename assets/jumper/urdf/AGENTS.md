# Instructions for this URDF directory

Read URDF_CONVENTIONS.md before generating or modifying robot assets under this directory. It records the user's accepted naming and export conventions. The user's latest explicit instruction takes precedence.

- Reference model: jumper/urdf/jumper.urdf. Robot/product name is Jumper; identifier is jumper.
- Keep LF/RF/LM/RM/LR/RR case. Movable joints use <PREFIX>_J<index>_joint, starting at J0 per limb (LF/RF: J0-J4; walking legs: J0-J2). Links and fixed joints keep semantic English names; current separate foot-tip pads retain foot_tip identifiers. No pinyin or Chinese release filenames/identifiers.
- Versions belong in the URDF's leading XML comment, not names, paths or release.json.
- Default collision policy: visual and collision share the same original STL, origin and scale. Do not reintroduce primitive fitting, decimation or separate collision meshes unless requested.
- Preserve joints, mass/inertia, colors and mesh payloads outside the requested changes. Base mass in the current reference is 0.886006064277726 kg.
- Keep the distributable minimal and complete. Do not add unrequested simulation frameworks, ROS scaffolding or reports.
- Use independent locations for experiments; do not overwrite the reference with a trial.
- Do not assume camera/motor YAML is active without checking its consumer.
- Verify changes at an appropriate scope and distinguish loading success from collision/dynamics validation.
- Use focused reads and appropriate model delegation to limit cost. Trivial work does not require a subagent.

See URDF_CONVENTIONS.md for exact naming tables, colors, paths, metadata format, physical baseline and validation requirements.
