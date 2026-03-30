"""
3D Augmentation Pipeline for Dermatological Images
Implements perspective warping, lighting simulation, and surface-aware transforms
to simulate different camera angles and conditions on skin lesions.
"""

import numpy as np
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter
from PIL import Image


class SurfaceNormalEstimator:
    """Estimate pseudo-3D surface normals from a 2D skin image."""

    @staticmethod
    def estimate(image_np):
        """
        Estimate surface normals using gradient-based approach.
        Returns normal map (H, W, 3) with x, y, z components.
        """
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float64)
        gray = gaussian_filter(gray, sigma=1.5)

        # Compute gradients
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        # Build normal map: (-dx, -dy, 1) normalized
        normals = np.zeros((*gray.shape, 3), dtype=np.float64)
        normals[..., 0] = -dx
        normals[..., 1] = -dy
        normals[..., 2] = 1.0

        # Normalize
        magnitude = np.linalg.norm(normals, axis=-1, keepdims=True)
        magnitude = np.clip(magnitude, 1e-8, None)
        normals /= magnitude

        return normals


class PerspectiveAugmentor:
    """Simulate 3D perspective changes as if viewing the skin from different angles."""

    @staticmethod
    def apply(image_np, pitch=0, yaw=0, roll=0, scale=1.0):
        """
        Apply 3D perspective warp.
        pitch: rotation around X axis (degrees)
        yaw: rotation around Y axis (degrees)
        roll: rotation around Z axis (degrees)
        """
        h, w = image_np.shape[:2]
        cx, cy = w / 2, h / 2

        # Convert to radians
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        roll_rad = np.radians(roll)

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)],
        ])
        Ry = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)],
        ])
        Rz = np.array([
            [np.cos(roll_rad), -np.sin(roll_rad), 0],
            [np.sin(roll_rad), np.cos(roll_rad), 0],
            [0, 0, 1],
        ])

        R = Rz @ Ry @ Rx

        # Focal length approximation
        f = max(w, h)

        # Source corners (centered)
        corners = np.array([
            [-cx, -cy, 0],
            [w - cx, -cy, 0],
            [w - cx, h - cy, 0],
            [-cx, h - cy, 0],
        ], dtype=np.float64)

        # Project through rotation
        projected = []
        for pt in corners:
            rotated = R @ pt
            # Perspective projection
            px = f * rotated[0] / (f + rotated[2]) + cx
            py = f * rotated[1] / (f + rotated[2]) + cy
            projected.append([px, py])

        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst_pts = np.array(projected, dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(image_np, M, (w, h),
                                     borderMode=cv2.BORDER_REFLECT_101)
        return result


class LightingAugmentor:
    """Simulate varying lighting conditions using surface normals."""

    @staticmethod
    def apply(image_np, normals, light_direction=None, ambient=0.3, diffuse=0.7):
        """
        Apply Lambertian lighting model using estimated surface normals.
        light_direction: (3,) unit vector for light direction
        """
        if light_direction is None:
            # Random light direction
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(np.pi / 6, np.pi / 3)
            light_direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ])

        light_direction = light_direction / np.linalg.norm(light_direction)

        # Lambertian shading
        intensity = np.sum(normals * light_direction, axis=-1)
        intensity = np.clip(intensity, 0, 1)

        # Combined lighting
        shading = ambient + diffuse * intensity
        shading = np.clip(shading, 0, 1)

        # Apply to image
        result = image_np.astype(np.float64)
        result *= shading[..., np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result


class ElasticDeformation:
    """Simulate skin surface deformation (stretching, compression)."""

    @staticmethod
    def apply(image_np, alpha=80, sigma=10):
        """Apply elastic deformation to simulate skin surface variation."""
        h, w = image_np.shape[:2]

        dx = gaussian_filter(np.random.randn(h, w) * alpha, sigma)
        dy = gaussian_filter(np.random.randn(h, w) * alpha, sigma)

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        result = cv2.remap(image_np, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
        return result


class ColorAugmentor:
    """Augment color to simulate different skin tones and camera white balance."""

    @staticmethod
    def apply(image_np, hue_shift=0, sat_scale=1.0, val_scale=1.0):
        """Apply HSV-space color augmentation."""
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype(np.float64)
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


class DermatologyAugmentor3D:
    """
    Full 3D augmentation pipeline for dermatological images.
    Combines perspective warping, surface-aware lighting, elastic deformation,
    and color augmentation.
    """

    def __init__(self, severity="medium"):
        self.surface_estimator = SurfaceNormalEstimator()
        self.perspective = PerspectiveAugmentor()
        self.lighting = LightingAugmentor()
        self.elastic = ElasticDeformation()
        self.color = ColorAugmentor()

        severity_params = {
            "light": {"angle": 10, "alpha": 40, "hue": 5, "sat": 0.1, "val": 0.1},
            "medium": {"angle": 20, "alpha": 80, "hue": 10, "sat": 0.2, "val": 0.15},
            "heavy": {"angle": 35, "alpha": 120, "hue": 15, "sat": 0.3, "val": 0.2},
        }
        self.params = severity_params.get(severity, severity_params["medium"])

    def augment(self, image, return_steps=False):
        """
        Apply full 3D augmentation pipeline.
        image: PIL Image
        return_steps: if True, return intermediate augmentation steps
        """
        img_np = np.array(image)
        steps = {"original": img_np.copy()}

        # 1. Estimate surface normals
        normals = self.surface_estimator.estimate(img_np)
        steps["normals"] = ((normals + 1) / 2 * 255).astype(np.uint8)

        # 2. 3D Perspective warp
        pitch = np.random.uniform(-self.params["angle"], self.params["angle"])
        yaw = np.random.uniform(-self.params["angle"], self.params["angle"])
        roll = np.random.uniform(-self.params["angle"] / 2, self.params["angle"] / 2)
        img_np = self.perspective.apply(img_np, pitch=pitch, yaw=yaw, roll=roll)
        steps["perspective"] = img_np.copy()

        # Re-estimate normals after warp
        normals = self.surface_estimator.estimate(img_np)

        # 3. Surface-aware lighting
        img_np = self.lighting.apply(img_np, normals)
        steps["lighting"] = img_np.copy()

        # 4. Elastic deformation
        img_np = self.elastic.apply(img_np, alpha=self.params["alpha"])
        steps["elastic"] = img_np.copy()

        # 5. Color augmentation
        hue_shift = np.random.uniform(-self.params["hue"], self.params["hue"])
        sat_scale = np.random.uniform(1 - self.params["sat"], 1 + self.params["sat"])
        val_scale = np.random.uniform(1 - self.params["val"], 1 + self.params["val"])
        img_np = self.color.apply(img_np, hue_shift, sat_scale, val_scale)
        steps["color"] = img_np.copy()

        result = Image.fromarray(img_np)

        if return_steps:
            step_images = {k: Image.fromarray(v) for k, v in steps.items()}
            return result, step_images

        return result
