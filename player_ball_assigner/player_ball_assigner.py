import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    """Assigns the ball to the nearest player."""

    def __init__(self):
        """Initialize with max allowed player-ball distance."""
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        """Assign the ball to the closest player within max distance."""
        ball_position = get_center_of_bbox(ball_bbox)
        min_distance = sys.maxsize
        assigned_player = None

        for player_id, player in players.items():
            player_bbox = player['bbox']
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id

        return assigned_player
