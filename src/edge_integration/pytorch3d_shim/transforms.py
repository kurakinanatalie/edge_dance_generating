import torch
import math

__all__ = [
    "RotateAxisAngle",
    "axis_angle_to_quaternion",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "axis_angle_to_matrix",
    "matrix_to_axis_angle",
    "matrix_to_rotation_6d",
    "rotation_6d_to_matrix",
    "quaternion_conjugate",
    "quaternion_multiply",
    "quaternion_invert",
    "quaternion_apply",
]


def _normalize_quaternion(q, eps=1e-8):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def axis_angle_to_quaternion(aa: torch.Tensor) -> torch.Tensor:
    angle = aa.norm(dim=-1, keepdim=True)
    small = angle < 1e-8
    axis = torch.where(
        small,
        torch.tensor([1.0, 0.0, 0.0], device=aa.device, dtype=aa.dtype).expand_as(aa),
        aa / angle.clamp_min(1e-8),
    )
    half = angle * 0.5
    w = torch.cos(half)
    s = torch.sin(half)
    xyz = axis * s
    q = torch.cat([w, xyz], dim=-1)
    return _normalize_quaternion(q)


def quaternion_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    q = _normalize_quaternion(q)
    w, xyz = q[..., :1], q[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w.clamp_min(1e-8))
    axis = torch.where(
        sin_half < 1e-8,
        torch.tensor([1.0, 0.0, 0.0], device=q.device, dtype=q.dtype).expand_as(xyz),
        xyz / sin_half.clamp_min(1e-8),
    )
    return axis * angle


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = _normalize_quaternion(q)
    w, x, y, z = q.unbind(-1)
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)
    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)
    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)
    return torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    t = m00 + m11 + m22
    w = torch.sqrt(torch.clamp(t + 1.0, min=0.0)) / 2.0
    x = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)) / 2.0
    y = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0)) / 2.0
    z = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0)) / 2.0
    x = torch.copysign(x, m21 - m12)
    y = torch.copysign(y, m02 - m20)
    z = torch.copysign(z, m10 - m01)
    q = torch.stack([w, x, y, z], dim=-1)
    return _normalize_quaternion(q)


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(axis_angle_to_quaternion(aa))


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    return quaternion_to_axis_angle(matrix_to_quaternion(R))


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    a1 = R[..., :, 0]
    a2 = R[..., :, 1]
    return torch.cat([a1, a2], dim=-1)


def rotation_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    a1 = x[..., :3]
    a2 = x[..., 3:6]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    proj = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - proj * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1 = _normalize_quaternion(q1)
    q2 = _normalize_quaternion(q2)
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return _normalize_quaternion(torch.stack([w, x, y, z], dim=-1))


def quaternion_invert(q: torch.Tensor) -> torch.Tensor:
    return quaternion_conjugate(_normalize_quaternion(q))


def quaternion_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    R = quaternion_to_matrix(q)
    return (R @ v.unsqueeze(-1)).squeeze(-1)


class RotateAxisAngle:
    """
    Minimal replacement for pytorch3d.transforms.RotateAxisAngle.
    Provides get_matrix() returning a rotation matrix for a given axis/angle.
    """

    def __init__(self, angle, axis="X", degrees=True):
        if not torch.is_tensor(angle):
            angle = torch.tensor(angle, dtype=torch.float32)
        if degrees:
            angle = angle * math.pi / 180.0
        self.angle = angle
        self.axis = axis.upper()

    def get_matrix(self):
        a = self.angle
        c = torch.cos(a)
        s = torch.sin(a)
        if self.axis == "X":
            R = torch.stack(
                [
                    torch.stack(
                        [
                            torch.ones_like(c),
                            torch.zeros_like(c),
                            torch.zeros_like(c),
                        ],
                        dim=-1,
                    ),
                    torch.stack([torch.zeros_like(c), c, -s], dim=-1),
                    torch.stack([torch.zeros_like(c), s, c], dim=-1),
                ],
                dim=-2,
            )
        elif self.axis == "Y":
            R = torch.stack(
                [
                    torch.stack([c, torch.zeros_like(c), s], dim=-1),
                    torch.stack(
                        [
                            torch.zeros_like(c),
                            torch.ones_like(c),
                            torch.zeros_like(c),
                        ],
                        dim=-1,
                    ),
                    torch.stack([-s, torch.zeros_like(c), c], dim=-1),
                ],
                dim=-2,
            )
        else:
            R = torch.stack(
                [
                    torch.stack([c, -s, torch.zeros_like(c)], dim=-1),
                    torch.stack([s, c, torch.zeros_like(c)], dim=-1),
                    torch.stack(
                        [
                            torch.zeros_like(c),
                            torch.zeros_like(c),
                            torch.ones_like(c),
                        ],
                        dim=-1,
                    ),
                ],
                dim=-2,
            )
        return R