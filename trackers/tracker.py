from __future__ import annotations

import os
import sys
import pickle
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv

sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    """
    Tracker class for detecting and tracking players, referees, and the ball
    """

    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks: Dict[str, List[Dict[int, Dict[str, Any]]]]) -> None:
        """
        Add computed positions (center or foot) to object tracks.
        """
        for obj_type, object_tracks in tracks.items():
            for frame_tracks in object_tracks:
                for track_id, track_info in frame_tracks.items():
                    bbox = track_info["bbox"]
                    position = (
                        get_center_of_bbox(bbox) if obj_type == "ball" else get_foot_position(bbox)
                    )
                    track_info["position"] = position

    def interpolate_ball_positions(
        self, ball_positions: List[Dict[int, Dict[str, List[float]]]]
    ) -> List[Dict[int, Dict[str, Dict[str, List[float]]]]]:
        """
        Interpolates missing ball positions for smoother tracking.
        """
        raw_positions = [frame.get(1, {}).get("bbox", []) for frame in ball_positions]
        df_positions = pd.DataFrame(raw_positions, columns=["x1", "y1", "x2", "y2"])
        df_positions = df_positions.interpolate().bfill()

        return [{1: {"bbox": bbox.tolist()}} for bbox in df_positions.to_numpy()]

    def detect_frames(self, frames: List[np.ndarray], batch_size: int = 20) -> List[Any]:
        """
        Run YOLO detection on a list of frames in batches.
        """
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            detections.extend(self.model.predict(batch, conf=0.1))
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    # ----------------------------------------------------------------------
    # Drawing Utilities
    # ----------------------------------------------------------------------

    def draw_ellipse(
        self, frame: np.ndarray, bbox: List[float], color: tuple[int, int, int], track_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Draw an ellipse (player/referee marker) on a frame.
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(width * 0.35)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        return frame

    def draw_triangle(self, frame: np.ndarray, bbox: List[float], color: tuple[int, int, int]) -> np.ndarray:
        """
        Draw a filled triangle marker on top of an object (ball/player with ball).
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        pts = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(
        self, frame: np.ndarray, frame_num: int, team_ball_control: np.ndarray
    ) -> np.ndarray:
        """
        Draw ball possession statistics overlay on the frame.
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        control_so_far = team_ball_control[: frame_num + 1]
        t1_frames = (control_so_far == 1).sum()
        t2_frames = (control_so_far == 2).sum()
        total = t1_frames + t2_frames

        if total > 0:
            t1_ratio, t2_ratio = t1_frames / total, t2_frames / total
        else:
            t1_ratio = t2_ratio = 0.0

        cv2.putText(
            frame,
            f"Team 1 ball control : {t1_ratio * 100:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Team 2 ball control : {t2_ratio * 100:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        return frame

    def draw_annotations(
        self,
        video_frames: List[np.ndarray],
        tracks: Dict[str, List[Dict[int, Dict[str, Any]]]],
        team_ball_control: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Draw all annotations (players, referees, ball, and possession stats) on video frames.
        """
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            for track_id, player in tracks["players"][frame_num].items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            for _, referee in tracks["referees"][frame_num].items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for _, ball in tracks["ball"][frame_num].items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_frames.append(frame)

        return output_frames