from typing import Dict, Tuple, Any
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """
    Class responsible for assigning players to teams based on their dominant jersey colors.
    Utilizes KMeans clustering to determine color clusters for players and assign them to teams.
    """

    def __init__(self) -> None:
        self.kmeans: KMeans | None = None
        self.team_colors: Dict[int, np.ndarray] = {}
        self.player_team_dict: Dict[int, int] = {}

    @staticmethod
    def _fit_kmeans(image: np.ndarray) -> KMeans:
        """
        Fit a KMeans model on a given image reshaped into a 2D array of RGB pixels.
        """
        image_2d = image.reshape((-1, 3))
        model = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=1,
            random_state=0
        )
        model.fit(image_2d)
        return model

    def _extract_player_color(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Extract the dominant player color from the top half of the bounding box.
        """
        x1, y1, x2, y2 = map(int, bbox)
        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            # If bbox is invalid or out of frame, return a neutral color
            return np.array([0, 0, 0], dtype=np.float32)

        top_half = cropped[:cropped.shape[0] // 2, :]

        kmeans = self._fit_kmeans(top_half)
        labels = kmeans.labels_.reshape(top_half.shape[0], top_half.shape[1])

        # Assume corners represent the background (non-player)
        corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        non_player_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_colors(self, frame: np.ndarray, player_detections: Dict[int, Dict[str, Any]]) -> None:
        """
        Assign team colors by clustering player jersey colors.
        """
        player_colors = [
            self._extract_player_color(frame, detection["bbox"])
            for detection in player_detections.values()
        ]

        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        self.kmeans.fit(player_colors)

        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]

    def get_player_team(self, frame: np.ndarray, player_bbox: Tuple[float, float, float, float], player_id: int) -> int:
        """
        Get the team assignment for a player, caching the result.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self._extract_player_color(frame, player_bbox)
        team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1

        self.player_team_dict[player_id] = team_id
        return team_id
