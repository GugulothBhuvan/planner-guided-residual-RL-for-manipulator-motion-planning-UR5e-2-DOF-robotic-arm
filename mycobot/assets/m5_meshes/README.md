# M5 shell meshes (visual only)

STL conversions of the MyCobot 280 M5 shell meshes (`joint1`..`joint7`, `G_base`) from the Elephant Robotics
`mycobot_ros` description package (`mycobot_description/urdf/mycobot_280_m5/*.dae`,
https://github.com/elephantrobotics/mycobot_ros). Converted with trimesh; `G_base` is in metres like the others.
The licence of that repository was not checked here; check it before redistributing.

Used only for drawing, by `scripts/render_mesh_compare.py`: a separate visual-only MuJoCo model with the same bodies is
rendered from the simulation state, so physics, collisions, limits and the policy are unchanged (episode 8: identical
trajectory and outcome with and without the meshes). Mesh placement follows the visual origins of the official M5 URDF
(`assets/mycobot_280_m5.urdf`). The suction head has no mesh (the pump-head file was not in the downloaded folder).
