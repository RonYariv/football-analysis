import numpy as np
import cv2


class ViewTransformer:
    """Transforms player positions from pixel space to real-world court coordinates."""

    def __init__(self):
        """Initialize the perspective transformation using known court dimensions."""
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ])

        # Real-world coordinates of the same corners in meters
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        # Convert to float32 for OpenCV compatibility
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Compute perspective transform matrix
        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point):
        """Transform a single pixel point into real-world coordinates."""
        p = (int(point[0]), int(point[1]))

        # Ensure the point lies within the polygon of reference pixels
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        # Reshape and apply perspective transform
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(
            reshaped_point, self.perspective_transformer
        )
        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        """Add real-world transformed positions to tracked objects."""
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)

                    # Apply perspective transform
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()

                    # Store the transformed position in the track info
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed