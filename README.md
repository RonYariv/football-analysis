# Soccer Player Tracking & Analytics

A **highly advanced computer vision system** for tracking soccer players and the ball, analyzing movements, speeds, distances, team assignment, and ball possession from video footage.

---

## Features

This project implements a complete pipeline for **soccer match analysis**, combining deep learning, tracking, and advanced computer vision techniques:

- **Player and Ball Detection**  
  - Detects players, referees, and the ball in video frames using YOLO-based object detection.  
  - Handles goalkeeper class separately and merges into the player class.

- **Tracking Across Frames**  
  - Tracks each player and the ball frame-by-frame using ByteTrack.  
  - Maintains persistent track IDs for each object.

- **Camera Movement Compensation**  
  - Estimates camera motion using Lucas-Kanade optical flow.  
  - Adjusts all positions relative to the moving camera for accurate measurements.

- **Real-World Coordinate Transformation**  
  - Maps pixel coordinates to actual soccer field dimensions using perspective transformation.  
  - Enables precise player position and movement tracking on the field.

- **Speed and Distance Calculation**  
  - Calculates speed (km/h) and cumulative distance (meters) for each player.  
  - Provides per-frame metrics visible in video annotations.

- **Team Assignment**  
  - Automatically identifies team membership based on jersey color clustering.
  
- **Ball Possession Tracking**  
  - Assigns ball possession to the closest player intelligently.  
  - Computes team ball control statistics over time.

- **Annotated Output Video**  
  - Draws ellipses around players, triangles for ball possession, and overlays speed, distance, and team control.  
  - Generates a visually informative output video for analysis.

---
