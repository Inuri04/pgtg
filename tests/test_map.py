import pytest

from pgtg.map import EpisodeMap
from pgtg.parser import json_file_to_map_plan


class TestGetterAndSetter:
    class TestGetFeaturesAt:
        def test_arguments_out_of_bound_raise_error(self, map_1x1):
            map = map_1x1

            with pytest.raises(ValueError):
                map.get_features_at(-1, 0)

            with pytest.raises(ValueError):
                map.get_features_at(0, -1)

            with pytest.raises(ValueError):
                map.get_features_at(9, 0)

            with pytest.raises(ValueError):
                map.get_features_at(0, 9)

    class TestSetFeaturesAt:
        def test_arguments_out_of_bound_raise_error(self, map_1x1):
            map = map_1x1

            with pytest.raises(ValueError):
                map.set_features_at(-1, 0, {"wall"})

            with pytest.raises(ValueError):
                map.set_features_at(0, -1, {"wall"})

            with pytest.raises(ValueError):
                map.set_features_at(9, 0, {"wall"})

            with pytest.raises(ValueError):
                map.set_features_at(0, 9, {"wall"})


class TestTrafficLanes:
    def test_store_car_spawners(self, map_with_all_deadends):
        map = map_with_all_deadends

        assert len(map.car_spawners) == 6

        # start spawner
        assert (0, 36 + 5) in map.car_spawners
        # goal spawner
        assert (36 + 8, 3) in map.car_spawners

        # deadend north
        assert (36 + 5, 9 + 5) in map.car_spawners
        # deadend east
        assert (9 + 3, 18 + 5) in map.car_spawners
        # deadend south
        assert (3, 27 + 3) in map.car_spawners
        # deadend west
        assert (27 + 5, 18 + 3) in map.car_spawners

    def test_store_traffic_spawnable_positions(self, map_with_all_deadends):
        map = map_with_all_deadends

        # spawnable positions on the start line
        assert (0, 39) in map.traffic_spawnable_positions
        assert (0, 41) in map.traffic_spawnable_positions

        # spawnable positions on straight tile
        assert (9, 39) in map.traffic_spawnable_positions
        assert (9, 41) in map.traffic_spawnable_positions
        assert (13, 39) in map.traffic_spawnable_positions
        assert (13, 41) in map.traffic_spawnable_positions
        assert (17, 39) in map.traffic_spawnable_positions
        assert (17, 41) in map.traffic_spawnable_positions

        # spawnable positions on turn tile (inner turn)
        assert (18, 39) in map.traffic_spawnable_positions
        assert (20, 39) in map.traffic_spawnable_positions
        assert (20, 38) in map.traffic_spawnable_positions
        assert (21, 38) in map.traffic_spawnable_positions
        assert (21, 37) in map.traffic_spawnable_positions

        # spawnable positions on turn tile (outer turn)
        assert (18, 41) in map.traffic_spawnable_positions
        assert (22, 41) in map.traffic_spawnable_positions
        assert (22, 40) in map.traffic_spawnable_positions
        assert (23, 40) in map.traffic_spawnable_positions
        assert (23, 39) in map.traffic_spawnable_positions
        assert (23, 36) in map.traffic_spawnable_positions

        # spawnable positions on crossing tile (exits)
        assert (18, 21) in map.traffic_spawnable_positions
        assert (18, 23) in map.traffic_spawnable_positions
        assert (26, 21) in map.traffic_spawnable_positions
        assert (26, 23) in map.traffic_spawnable_positions
        assert (21, 18) in map.traffic_spawnable_positions
        assert (23, 18) in map.traffic_spawnable_positions
        assert (21, 26) in map.traffic_spawnable_positions
        assert (23, 26) in map.traffic_spawnable_positions

        # spawnable positions on crossing tile (corners)
        assert (20, 20) in map.traffic_spawnable_positions
        assert (24, 20) in map.traffic_spawnable_positions
        assert (20, 24) in map.traffic_spawnable_positions
        assert (24, 24) in map.traffic_spawnable_positions

        # spawnable positions on crossing tile (middle)
        assert (21, 21) in map.traffic_spawnable_positions
        assert (22, 21) in map.traffic_spawnable_positions
        assert (23, 21) in map.traffic_spawnable_positions
        assert (21, 22) in map.traffic_spawnable_positions
        assert (22, 22) in map.traffic_spawnable_positions
        assert (23, 22) in map.traffic_spawnable_positions
        assert (21, 23) in map.traffic_spawnable_positions
        assert (22, 23) in map.traffic_spawnable_positions
        assert (23, 23) in map.traffic_spawnable_positions

        # spawnable positions on the goal line
        assert (44, 3) in map.traffic_spawnable_positions
        assert (44, 5) in map.traffic_spawnable_positions


@pytest.fixture
def map_1x1() -> EpisodeMap:
    map_object = json_file_to_map_plan("tests/test_data/1x1_map")
    return EpisodeMap(map_object)


@pytest.fixture
def map_with_all_deadends() -> EpisodeMap:
    map_object = json_file_to_map_plan("tests/test_data/map_with_all_deadends")
    return EpisodeMap(map_object)
