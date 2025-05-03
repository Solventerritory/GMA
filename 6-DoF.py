import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
from scipy.spatial.transform import Rotation

def generate_orientation_keypoints(joint_keypoint, bone_length, bone_transform):
  """
  Generates the four orientation keypoints (OKPs) for a given bone.

  Args:
    joint_keypoint:  3D coordinates of the parent joint (x, y, z) - numpy array of shape (3,)
    bone_length:     Length of the bone (scalar)
    bone_transform:  4x4 transformation matrix (rotation and translation)
                     from bone's keyframe to world space.
                     Can be represented as a numpy array.

  Returns:
    okps:            A 4x3 numpy array containing the 3D coordinates of the four OKPs.
  """

  l_k = bone_length
  T_k = bone_transform

  # Define offsets in the bone's coordinate system
  offsets = np.array([
      [0.5, 0.5, 0],
      [0.5, -0.5, 0],
      [0.5, 0, 0.5],
      [0.5, 0, -0.5]
  ]).T  # Shape: (3, 4)

  #  Apply transformation and scaling to get OKP coordinates
  okps = joint_keypoint.reshape(3, 1) + T_k[:3,:3] @ (l_k * offsets)

  return okps.T  # Shape: (4, 3)


class CrosshairsNet(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.head_x = CrosshairsHead1D(2048, num_keypoints)
        self.head_y = CrosshairsHead1D(2048, num_keypoints)
        self.head_z = CrosshairsHead1D_Z(2048, num_keypoints)

    def forward(self, x):
        c5 = self.backbone(x) #  Get C5 feature map
        x_out = self.head_x(c5)
        y_out = self.head_y(c5)
        z_out = self.head_z(c5)
        return torch.stack([x_out, y_out, z_out], dim=-1) # (Batch_size, Num_keypoints, 3)

class CrosshairsHead1D(nn.Module):
  def __init__(self, in_channels, num_keypoints):
    super().__init__()
    self.conv_init = nn.Conv2d(in_channels, 256, kernel_size=1)
    self.conv_flatten = nn.Conv2d(256, 256, kernel_size=(8, 1)) #  Flatten y
    self.bottleneck = nn.Conv2d(256, 256, kernel_size=(1, 9), padding=(0, 4))
    self.conv_transpose = nn.ConvTranspose2d(256, 256, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))
    self.conv_final = nn.Conv2d(256, num_keypoints, kernel_size=1)

  def forward(self, x):
    x = self.conv_init(x)
    x = self.conv_flatten(x)
    x = self.bottleneck(x)
    for _ in range(5):
      x = self.conv_transpose(x)
    x = self.conv_final(x)
    return x.squeeze(1) #  (Batch_size, Num_keypoints, Width)

class CrosshairsHead1D_Z(nn.Module):
  def __init__(self, in_channels, num_keypoints):
    super().__init__()
    self.conv_init = nn.Conv2d(in_channels, 256 * 12, kernel_size=1)
    self.conv_reshape = nn.Conv3d(256, 256, kernel_size=(9, 9, 12))
    self.conv_flatten = nn.Conv3d(256, 256, kernel_size=(1, 9, 12))
    self.conv_transpose = nn.ConvTranspose3d(256, 256, kernel_size=(1, 1, 4), stride=(1, 1, 2), padding=(0, 0, 1))
    self.conv_final = nn.Conv2d(256, num_keypoints, kernel_size=1)

  def forward(self, x):
    x = self.conv_init(x)
    x = x.reshape(-1, 256, 9, 9, 12)
    x = self.conv_reshape(x)
    x = self.conv_flatten(x)
    for _ in range(5):
      x = self.conv_transpose(x)
    x = x.squeeze(2).squeeze(2) #  (Batch_size, Num_keypoints, Depth)
    x = self.conv_final(x)
    return x.squeeze(1)

def coord_weighted_softmax(heatmaps):
  """
  Calculates the keypoint coordinates from 1D heatmaps using a weighted softmax.

  Args:
    heatmaps:  Tensor of shape (Batch_size, Num_keypoints, Length)

  Returns:
    coordinates: Tensor of shape (Batch_size, Num_keypoints)
  """
  batch_size, num_keypoints, length = heatmaps.shape
  weights = torch.linspace(-1, 1, length).to(heatmaps.device)

  softmax_out = torch.softmax(heatmaps, dim=-1)

  coordinates = (softmax_out * weights).sum(dim=-1) / softmax_out.sum(dim=-1)
  return coordinates

def calculate_bone_rotation(joint_keypoint, orientation_keypoints, bone_length, neutral_pose_data):
    """
    Calculates the rotation of a bone from the predicted keypoints.

    Args:
        joint_keypoint:         3D coordinates of the joint (parent) - (3,)
        orientation_keypoints:  3D coordinates of the 4 OKPs  - (4, 3)
        bone_length:            Length of the bone
        neutral_pose_data:    Dictionary containing the neutral pose positions
                              of the joint and OKPs, normalized by bone length.
                              { 'joint': (3,), 'okp1': (3,), 'okp2': (3,), 'okp3':(3,), 'okp4': (3,) }

    Returns:
        rotation_matrix:      3x3 rotation matrix
    """

    #  1.  Recover 3D points in "bone space" (centered at joint, normalized by length)
    bone_points_predicted = (np.concatenate([orientation_keypoints, joint_keypoint.reshape(1,3)]) - joint_keypoint) / bone_length
    bone_points_neutral = np.array([
        neutral_pose_data['okp1'],
        neutral_pose_data['okp2'],
        neutral_pose_data['okp3'],
        neutral_pose_data['okp4'],
        neutral_pose_data['joint']
    ])

    #  2. Solve for the rotation using least squares (Procrustes analysis or similar)
    #  cv2.estimateAffine3D  or  scipy.linalg.orthogonal_procrustes  are options
    _, rotation_matrix = cv2.estimateAffine3D(bone_points_neutral.reshape(-1, 1, 3).astype(np.float64),
                                            bone_points_predicted.reshape(-1, 1, 3).astype(np.float64),
                                            ransacThreshold=10) # You might need to tune ransacThreshold

    return rotation_matrix[:3, :3] if rotation_matrix is not None else np.eye(3)

if __name__ == '__main__':
    # Example Usage of Orientation Keypoint Generation
    joint_pos = np.array([1.0, 2.0, 3.0])
    bone_len = 10.0
    bone_pose_matrix = np.array([
        [0.8, -0.6, 0, 1.0],
        [0.6, 0.8, 0, 2.0],
        [0,    0,   1, 3.0],
        [0,    0,   0, 1  ]
    ]) # Example 4x4 transformation

    orientation_kpts = generate_orientation_keypoints(joint_pos, bone_len, bone_pose_matrix)
    print("Orientation Keypoints:\n", orientation_kpts)

    # Example Instantiation of Crosshairs Network
    num_joints = 17
    num_okps_per_joint = 4
    total_keypoints = num_joints + num_joints * num_okps_per_joint  # 17 + 17*4 = 85

    net = CrosshairsNet(total_keypoints)
    print("\nCrosshairs Network Architecture:")
    print(net)

    # Example Usage of Rotation Calculation
    neutral_pose = {
        'joint': np.array([0.0, 0.0, 0.0]),
        'okp1': np.array([0.5, 0.5, 0.0]),
        'okp2': np.array([0.5, -0.5, 0.0]),
        'okp3': np.array([0.5, 0.0, 0.5]),
        'okp4': np.array([0.5, 0.0, -0.5])
    }

    predicted_rotation = calculate_bone_rotation(joint_pos, orientation_kpts, bone_len, neutral_pose)
    print("\nPredicted Rotation Matrix:\n", predicted_rotation)

    # Convert to Euler Angles (if needed)
    euler_angles = Rotation.from_matrix(predicted_rotation).as_euler('xyz', degrees=True)
    print("Euler Angles (XYZ):", euler_angles)

    # Example of coord_weighted_softmax (dummy heatmap)
    dummy_heatmaps_x = torch.randn(2, total_keypoints, 64)
    predicted_x_coords = coord_weighted_softmax(dummy_heatmaps_x)
    print("\nPredicted X Coordinates from Heatmaps:\n", predicted_x_coords)
