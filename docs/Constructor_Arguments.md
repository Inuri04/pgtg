# Environment Arguments

The PGTG environment is highly customizable. Use these constructor arguments to modify it for your use case.

| Argument | Type | Default Value | Meaning |
|---|---|---|---|
|`map_path`|`str` or `None`|`None`|Path to a pregenerated map to use. If set to `None` a new random map is generated instead.|
|`random_map_width`|`int`|`4`|How many tiles the random generated map is wide.|
|`random_map_height`|`int`|`4`|How many tiles the random generated map is heigh.|
|`random_map_percentage_of_connections`|`float`|`0.5`|How many of the possible connections between tiles are generated in the random map. Each map has a path from start to goal, so a value of `0.0` will result in minimal connections, not none.|
|`random_map_start_position`|`tuple[int, int]` or `tuple[int, int, str]` or `"random"`|`(0, -1, "west")`|Where the start of the random generated map is located. The two integers specify the x and y coordinates of the tile the start is located in, optionally its direction ("north", "east", "south", or "west") can be specified. The start can only be located on a map border.|
|`random_map_goal_position`|`tuple[int, int]` or `tuple[int, int, str]` or `"random"`|`(-1, 0, "east")`|Where the goal of the random generated map is located. The two integers specify the x and y coordinates of the tile the goal is located in, optionally its direction ("north", "east", "south", or "west") can be specified. The goal can only be located on a map border.|
|`random_map_minimum_distance_between_start_and_goal`|`int` or `none`|`None`|The minimal length of the path between start and goal. Can only be used if both `random_map_start_position` and `random_map_goal_position` are set to `"random"`. The maximal allowed value is the hight plus the width of the map minus 2.|
|`random_map_obstacle_probability`|`float` in [0,1]|`0.0`|The probability for each tile of the random map to be generated with an obstacle.|
|`random_map_ice_probability_weight`|`float`|`1.0`|The relative probability of ice being chosen when an obstacle is generated for the random map.|
|`random_map_broken_road_probability_weight`|`float`|`1.0`|The relative probability of broken road being chosen when an obstacle is generated for the random map.|
|`random_map_sand_probability_weight`|`float`|`1.0`|The relative probability of sand being chosen when an obstacle is generated for the random map.|
|`random_map_traffic_light_probability_weight`|`float`|`1.0`|The relative probability of traffic lights being chosen when an obstacle is generated for the random map.|
|`render_mode`|`"human"` or `"pil_image"` or `None`|`None`|The [Gymnasium render mode](https://gymnasium.farama.org/api/env/#gymnasium.Env.render). If `"human"` is chosen a pygame window opens and advances to the next frame whenever `step()` is called, `"pil_image"` makes `render()` return a `PIL.Image.Image` and `None` to no output being generated.|
|`features_to_include_in_observation`|`list[str]`|`["walls", "goals", "ice", "broken road", "sand", "traffic", "traffic_light_green", "traffic_light_yellow", "traffic_light_red"]`|What features are included in the observation as one-hot encodings. Changing this argument changes the observation space.|
|`use_sliding_observation_window`|`bool`|`False`|Wether or not the part of the map that is observable should slide along centered on the agent. If set to `False` the tile the agent is currently in is observed. Changing this argument changes the observation space.|
|`sliding_observation_window_size`|`int`|`4`|How many squares the sliding observation window extends in all directions. A value of `1` results in a 3 x 3 observation windows, `2` in 5 x 5, and `3` in 7 x 7. Has no effect if `use_sliding_observation_window` is `False`.  Changing this argument changes the observation space.|
|`use_next_subgoal_direction`|`bool`|`False`|Wether or not to include a additional observation that indicates the direction of the next (sub) goal in the observation. Changing this argument changes the observation space.|
|`sum_subgoals_reward`|`int`|`100`|The total reward awarded for reaching all (sub) goals. It is equally divided among all (sub) goals.|
|`final_goal_bonus`|`int`|`0`|Additional reward for reaching the final goal in addition to the normal (sub) goal reward defined by `sum_subgoals_reward`.|
|`crash_penalty`|`int`|`100`|Penalty for moving into a wall or traffic. The value is subtracted from the reward, thus a positive value should be used.|
|`traffic_light_violation_penalty`|`int`|`50`|Penalty for running a red light. The value is subtracted from the reward, thus a positive value should be used.|
|`standing_still_penalty`|`int`|`0`|Penalty for not moving or accelerating each. It is applied each step. The value is subtracted from the reward, thus a positive value should be used.|
|`already_visited_position_penalty`|`int`|`0`|Penalty for moving to a square that was already visited this episode. The value is subtracted from the reward, thus a positive value should be used.|
|`ice_probability`|`float` in [0,1]|`0.1`|The probability of the ice obstacle triggering when moved onto and moving the agent in a random direction.|
|`street_damage_probability`|`float` in [0,1]|`0.1`|The probability of the broken road obstacle triggering when moved onto and destroying the agents tires (thus permanently limiting its to speed to 1).|
|`sand_probability`|`float` in [0,1]|`0.2`|The probability of the sand obstacle triggering when moved onto and halting the agent.|
|`traffic_density`|`float` in [0,1]|`0.0`|How many percent of the squares that could have traffic are occupied by it. A value of `1.0` means that all lanes are permanently completely filled with traffic, making it impossible to avoid collisions. Values between `0.01` and `0.05` are recommended.|
|`traffic_light_phases_duration`|`tuple[int, int, int]`|`(10, 3, 10)`|Duration of the traffic light phases (green, yellow, red) in steps. The second value can be set to `0` to disable the yellow light phase.|
|`ignore_traffic_collisions`|`bool`|`False`|Wether to ignore collisions with traffic. Sometimes useful for testing.|