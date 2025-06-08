#########################################
# Step 4: Lens Flare Adversarial Attack
#########################################
import cv2
import os
import numpy as np
from tqdm import tqdm 
def add_lens_ghosting_effect(image, light_pos, num_ghosts=6, base_max_radius=150, base_min_radius=30, 
                             base_intensity=1.2, size_decay_factor=0.8, spread_factor=1.4, 
                             ring_thickness=3, num_sparkles=10):
    """
    Adds a golden cinematic lens flare with:
    - warm colored ghosts,
    - starburst,
    - circular flare ring,
    - small bright circular sparkles.
    """
    
    h, w = image.shape[:2]
    
    # Scale the maximum and minimum radius based on the image size
    max_radius = int(base_max_radius * min(h, w) / 800)  # Base size is set for 800x800
    min_radius = int(base_min_radius * min(h, w) / 800)  # Same scaling
    
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    center = np.array([w // 2, h // 2])
    light_pos_arr = np.array(light_pos)
    direction = center - light_pos_arr
    d_norm = np.linalg.norm(direction)
    direction = direction / d_norm if d_norm != 0 else np.array([0, 0])

    # Warm ghost color palette
    colors = [
        (1.0, 0.9, 0.6),  # warm yellow
        (1.0, 0.8, 0.4),  # golden
        (1.0, 0.6, 0.3),  # orange
        (0.8, 0.4, 0.2),  # dark orange
    ]

    # --- Ghosts ---
    for i in range(1, num_ghosts + 1):
        frac = (i / (num_ghosts + 1)) * spread_factor
        pos = light_pos_arr + frac * (center - light_pos_arr)
        pos = tuple(np.round(pos).astype(int))

        radius = int(max_radius * (size_decay_factor ** i))
        radius = max(radius, min_radius)
        intensity = base_intensity * (1 - frac)

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, pos, radius, intensity, -1)

        sigma = max(1, int(radius * 0.6))
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)

        color = colors[i % len(colors)]
        for c in range(3):
            overlay[:, :, c] += mask * color[c]

    # --- Starburst lines ---
    length = max(h, w)
    directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
    for dx, dy in directions:
        end_point = (int(light_pos[0] + dx * length), int(light_pos[1] + dy * length))
        start_point = (int(light_pos[0] - dx * length), int(light_pos[1] - dy * length))
        cv2.line(overlay, start_point, end_point, (1.0, 0.9, 0.6), 2)

    # --- Circular ring flare ---
    ring_mask = np.zeros((h, w), dtype=np.float32)
    ring_radius = int(np.linalg.norm(center - light_pos_arr) * 0.8)
    cv2.circle(ring_mask, tuple(center), ring_radius, 1.0, thickness=ring_thickness)
    ring_mask = cv2.GaussianBlur(ring_mask, (0, 0), sigmaX=6)
    for c in range(3):
        overlay[:, :, c] += ring_mask * 0.8 * colors[1][c]

    # --- Small sparkles / micro glares ---
    for i in range(num_sparkles):
        frac = np.random.uniform(0.1, 1.2)
        pos = light_pos_arr + frac * (center - light_pos_arr)
        pos += np.random.normal(scale=10, size=2)  # add tiny offset for realism
        pos = np.clip(np.round(pos).astype(int), 0, [w - 1, h - 1])

        sparkle_radius = np.random.randint(3, 8)
        sparkle_intensity = np.random.uniform(0.4, 0.9)
        sparkle_color = (1.0, 0.85, 0.5)

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, tuple(pos), sparkle_radius, sparkle_intensity, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1)

        for c in range(3):
            overlay[:, :, c] += mask * sparkle_color[c]

    # --- Final blur and blending ---
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=2)
    result = image.astype(np.float32) + overlay * 255.0
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result
