# -*- coding: utf-8 -*-
import torch
import math
import matplotlib.pyplot as plt
import os  # 添加这个导入

# 一些物理常量与常用函数
PI = torch.tensor(math.pi)
EPS = 1e-8

def safe_divide(a, b):
    """安全除法，避免除以0"""
    return a / (b + EPS)

def schlick_fresnel(cos_theta, f0):
    """Schlick 菲涅尔近似，支持向量和张量运算"""
    return f0 + (1.0 - f0) * torch.pow(1.0 - cos_theta, 5.0)

def trowbridge_reitz_distribution(n_dot_h, alpha_x, alpha_y):
    """
    Trowbridge-Reitz（GGX）微表面分布函数，支持各向异性
    n_dot_h: 法线与半程向量的点积
    alpha_x: x 方向粗糙度相关参数
    alpha_y: y 方向粗糙度相关参数
    """
    alpha_x_sq = alpha_x * alpha_x
    alpha_y_sq = alpha_y * alpha_y
    denom = (n_dot_h * n_dot_h) * (alpha_x_sq * alpha_y_sq - 1.0) + 1.0
    return (alpha_x * alpha_y) / (PI * denom * denom)

def smith_ggx_geometry(n_dot_v, n_dot_l, alpha):
    """Smith 几何函数（GGX 形式），用于处理微表面遮挡"""
    alpha_sq = alpha * alpha
    lambda_v = n_dot_v + torch.sqrt(n_dot_v * n_dot_v * (1.0 - alpha_sq) + alpha_sq)
    lambda_l = n_dot_l + torch.sqrt(n_dot_l * n_dot_l * (1.0 - alpha_sq) + alpha_sq)
    return 1.0 / (lambda_v + lambda_l)

def compute_anisotropy_alphas(roughness, anisotropic):
    """根据各向异性参数计算 alpha_x 和 alpha_y"""
    aspect = torch.sqrt(1.0 - 0.9 * anisotropic)
    alpha = roughness * roughness
    alpha_x = torch.max(torch.tensor(0.001), alpha / aspect)
    alpha_y = torch.max(torch.tensor(0.001), alpha * aspect)
    return alpha_x, alpha_y

class DisneyPrincipledBSDF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 材质参数，可后续通过 set_parameters 灵活设置
        self.base_color = torch.tensor([1.0, 1.0, 1.0])
        self.subsurface = torch.tensor(0.0)
        self.metallic = torch.tensor(0.0)
        self.specular = torch.tensor(0.5)
        self.specular_tint = torch.tensor(0.0)
        self.roughness = torch.tensor(0.5)
        self.anisotropic = torch.tensor(0.0)
        self.sheen = torch.tensor(0.0)
        self.sheen_tint = torch.tensor(0.5)
        self.clearcoat = torch.tensor(0.0)
        self.clearcoat_roughness = torch.tensor(0.0)
        self.ior = torch.tensor(1.5)
        self.transmission = torch.tensor(0.0)
        self.transmission_roughness = torch.tensor(0.0)

    def set_parameters(self, **kwargs):
        """动态设置材质参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, (int, float)):
                    value = torch.tensor(value)
                setattr(self, key, value)

    def diffuse_term(self, n_dot_l, n_dot_v, v_dot_h):
        """改进的漫反射项，考虑能量守恒和次表面散射混合"""
        fl = schlick_fresnel(n_dot_l, torch.tensor(0.0))
        fv = schlick_fresnel(n_dot_v, torch.tensor(0.0))

        energy_bias = 0.5
        fd90 = energy_bias + 2.0 * v_dot_h * v_dot_h * self.roughness
        fd_l = 1.0 + (fd90 - 1.0) * fl
        fd_v = 1.0 + (fd90 - 1.0) * fv

        subsurface_scattering = 1.25 * (
            (1.0 - fl * fv / 2.0) -
            (1.0 - fl) * (1.0 - fv) * (1.0 / 3.0)
        )

        return self.base_color * (1.0 - self.metallic) * (
            (1.0 - self.subsurface) * fd_l * fd_v / PI +
            self.subsurface * subsurface_scattering / PI
        )

    def specular_term(self, n, v, l):
        """镜面反射项，包含各向异性、清漆层等处理"""
        h = (v + l) / torch.norm(v + l, dim=-1, keepdim=True)  # 半程向量
        n_dot_h = torch.clamp(torch.sum(n * h, dim=-1), 0.0, 1.0)
        n_dot_v = torch.clamp(torch.sum(n * v, dim=-1), 0.0, 1.0)
        n_dot_l = torch.clamp(torch.sum(n * l, dim=-1), 0.0, 1.0)
        v_dot_h = torch.clamp(torch.sum(v * h, dim=-1), 0.0, 1.0)

        # 各向异性处理，计算 alpha_x 和 alpha_y
        alpha_x, alpha_y = compute_anisotropy_alphas(self.roughness, self.anisotropic)

        # 计算 F0（基础反射率）
        dielectric_f0 = torch.tensor(0.04)
        base_f0 = torch.ones(3) * dielectric_f0
        tint = self.base_color / (torch.mean(self.base_color) + EPS)

        # 金属度混合
        metallic_f0 = self.base_color
        f0 = base_f0 * (1.0 - self.metallic) + metallic_f0 * self.metallic

        # 镜面色调调整
        if self.specular_tint > 0:
            f0 = f0 * (1.0 - self.specular_tint) + tint * self.specular_tint * f0

        # 菲涅尔项
        F = schlick_fresnel(v_dot_h, f0)

        # 微表面分布项（Trowbridge-Reitz）
        D = trowbridge_reitz_distribution(n_dot_h, alpha_x, alpha_y)

        # 几何函数项（Smith GGX）
        alpha_ggx = self.roughness * self.roughness
        G = smith_ggx_geometry(n_dot_v, n_dot_l, alpha_ggx)

        # 镜面反射 BRDF
        specular_brdf = (D * G * F) / torch.clamp(4.0 * n_dot_l * n_dot_v, EPS)

        # 清漆层处理
        if self.clearcoat > 0:
            clearcoat_f0 = torch.tensor(0.04)
            clearcoat_alpha = self.clearcoat_roughness * self.clearcoat_roughness
            clearcoat_D = trowbridge_reitz_distribution(n_dot_h, clearcoat_alpha, clearcoat_alpha)  # 清漆层默认各向同性
            clearcoat_G = smith_ggx_geometry(n_dot_v, n_dot_l, 0.25)  # 清漆层几何函数参数
            clearcoat_F = schlick_fresnel(v_dot_h, clearcoat_f0)
            clearcoat_brdf = (clearcoat_D * clearcoat_G * clearcoat_F) / torch.clamp(4.0 * n_dot_l * n_dot_v, EPS)
            specular_brdf += self.clearcoat * clearcoat_brdf

        return specular_brdf

    def sheen_term(self, v_dot_h):
        """布料等材质的边缘光泽项"""
        if self.sheen > 0:
            sheen_color = torch.ones(3) * (1.0 - self.sheen_tint) + self.base_color * self.sheen_tint
            return self.sheen * sheen_color * schlick_fresnel(v_dot_h, torch.tensor(0.0)) * (1.0 - self.metallic)
        return torch.zeros(3)

    def forward(self, n, v, l):
        """
        计算迪士尼原则性 BSDF 的总贡献
        n: 表面法线，形状 [N, 3]
        v: 视线方向，形状 [N, 3]
        l: 光线方向，形状 [N, 3]
        """
        # 归一化输入向量
        n = n / torch.norm(n, dim=-1, keepdim=True)
        v = v / torch.norm(v, dim=-1, keepdim=True)
        l = l / torch.norm(l, dim=-1, keepdim=True)

        h = (v + l) / torch.norm(v + l, dim=-1, keepdim=True)
        n_dot_v = torch.clamp(torch.sum(n * v, dim=-1), 0.0, 1.0)
        n_dot_l = torch.clamp(torch.sum(n * l, dim=-1), 0.0, 1.0)
        v_dot_h = torch.clamp(torch.sum(v * h, dim=-1), 0.0, 1.0)

        diffuse = self.diffuse_term(n_dot_l, n_dot_v, v_dot_h)
        specular = self.specular_term(n, v, l)
        sheen = self.sheen_term(v_dot_h)

        return diffuse + specular + sheen

class PathTracer:
    def __init__(self, width=400, height=400, max_depth=5):
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.bsdf = DisneyPrincipledBSDF()

    def render(self, scene, camera):
        """
        简单路径追踪渲染
        scene: 场景对象，需包含光线与物体相交的方法（这里简化，假设是球体场景示例）
        camera: 相机对象，包含生成射线的方法
        """
        image = torch.zeros((self.height, self.width, 3))
        for y in range(self.height):
            for x in range(self.width):
                # 生成相机射线
                ray = camera.generate_ray(x, y, self.width, self.height)
                color = self.trace_ray(ray, scene, self.max_depth)
                image[y, x] = color
        return image

    def trace_ray(self, ray, scene, depth):
        """递归追踪射线"""
        if depth == 0:
            return torch.zeros(3)
        # 这里简化，假设场景只有一个球体，实际应遍历场景中所有物体求交
        hit, n, p = scene.intersect_sphere(ray)
        if not hit:
            return torch.zeros(3)  # 背景色，这里设为黑色

        # 计算光线方向（假设用随机采样的间接光，这里简化为直接光示例）
        l = torch.tensor([0.5, 0.5, 1.0])  # 简单设置一个光源方向
        l = l / torch.norm(l)
        v = -ray.direction  # 视线方向是射线方向的反方向

        # 获取 BSDF 值
        bsdf_value = self.bsdf(n.unsqueeze(0), v.unsqueeze(0), l.unsqueeze(0))[0]

        # 直接光照贡献（简化，实际应采样多个光源或进行重要性采样）
        light_intensity = torch.clamp(torch.sum(n * l, dim=-1), 0.0, 1.0)
        direct_color = bsdf_value * light_intensity

        # 间接光照（这里简单递归，实际应做更好的采样，比如随机半球采样）
        # 随机生成一个半球方向作为反射方向示例
        new_dir = self.random_hemisphere_direction(n)
        new_ray = Ray(p, new_dir)
        indirect_color = self.trace_ray(new_ray, scene, depth - 1) * 0.5  # 简单衰减

        return direct_color + indirect_color

    def random_hemisphere_direction(self, n):
        """在法线半球内随机采样一个方向（简单实现）"""
        theta = torch.rand(1) * PI
        phi = torch.rand(1) * 2 * PI
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        dir = torch.tensor([x, y, z])
        # 确保在法线半球内
        if torch.dot(dir, n) < 0:
            dir = -dir
        return dir / torch.norm(dir)

# 辅助的射线和球体场景类（非常简化）
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

class SphereScene:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def intersect_sphere(self, ray):
        """射线与球体相交检测，返回是否命中、法线、交点"""
        oc = ray.origin - self.center
        a = torch.dot(ray.direction, ray.direction)
        b = 2 * torch.dot(oc, ray.direction)
        c = torch.dot(oc, oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return False, None, None
        t = (-b - torch.sqrt(discriminant)) / (2 * a)
        if t < 0:
            t = (-b + torch.sqrt(discriminant)) / (2 * a)
        if t < 0:
            return False, None, None
        p = ray.origin + t * ray.direction
        n = (p - self.center) / self.radius
        return True, n, p


class Camera:
    def __init__(self, eye, lookat, up, fov=90):
        self.eye = eye  # 相机位置（张量）
        self.lookat = lookat  # 目标点（张量）
        self.up = up  # 上方向量（张量）

        # 1. 计算相机坐标系的三个轴（forward, right, up）
        self.forward = (lookat - eye) / torch.norm(lookat - eye)  # 前向向量
        self.right = torch.cross(self.forward, up, dim=0)  # 右方向量（指定dim=0消除警告）
        self.right = self.right / torch.norm(self.right)  # 归一化
        self.up = torch.cross(self.right, self.forward, dim=0)  # 修正上方向量（与前向、右方垂直）
        self.up = self.up / torch.norm(self.up)  # 归一化

        # 2. 计算fov_scale（关键：确保正确定义该属性）
        self.fov = torch.tensor(fov, dtype=torch.float32)  # 将fov转为张量
        self.fov_scale = torch.tan(torch.deg2rad(self.fov / 2))  # 视场角缩放因子

    def generate_ray(self, x, y, width, height):
        """生成相机射线"""
        # 将像素坐标转换为 [-1, 1] 范围
        x_norm = (2 * (x + 0.5) / width) - 1
        y_norm = 1 - (2 * (y + 0.5) / height)
        # 计算射线方向
        dir = self.forward + self.right * x_norm * self.fov_scale + self.up * y_norm * self.fov_scale
        dir = dir / torch.norm(dir)
        return Ray(self.eye, dir)

if __name__ == "__main__":
    # 设置相机
    eye = torch.tensor([0.0, 0.0, -2.0])
    lookat = torch.tensor([0.0, 0.0, 0.0])
    up = torch.tensor([0.0, 1.0, 0.0])
    camera = Camera(eye, lookat, up, fov=45)

    # 设置场景（一个球体）
    sphere_center = torch.tensor([0.0, 0.0, 0.0])
    sphere_radius = torch.tensor(0.5)
    scene = SphereScene(sphere_center, sphere_radius)

    # 测试不同材质
    def test_different_materials(output_dir=None):
        materials = [
            {
                "name": "Polished Copper",
                "base_color": torch.tensor([0.95, 0.64, 0.54]),
                "metallic": torch.tensor(1.0),
                "roughness": torch.tensor(0.05),
                "specular_level": torch.tensor(0.5),
                "clearcoat": torch.tensor(0.0)
            },
            {
                "name": "Matte Plastic",
                "base_color": torch.tensor([0.2, 0.6, 0.8]),
                "metallic": torch.tensor(0.0),
                "roughness": torch.tensor(0.8),
                "specular_level": torch.tensor(0.5),
                "sheen": torch.tensor(0.2)
            },
            {
                "name": "Glass",
                "base_color": torch.tensor([0.98, 0.98, 0.98]),
                "metallic": torch.tensor(0.0),
                "roughness": torch.tensor(0.0),
                "specular_level": torch.tensor(0.5),
                "transmission": torch.tensor(1.0),
                "ior": torch.tensor(1.5),
                "transmission_roughness": torch.tensor(0.0)
            },
            {
                "name": "Skin",
                "base_color": torch.tensor([0.85, 0.65, 0.6]),
                "metallic": torch.tensor(0.0),
                "roughness": torch.tensor(0.4),
                "subsurface": torch.tensor(0.6),
                "sheen": torch.tensor(0.2),
                "sheen_tint": torch.tensor(0.5)
            },
            {
                "name": "Rough Aluminum",
                "base_color": torch.tensor([0.9, 0.9, 0.92]),
                "metallic": torch.tensor(1.0),
                "roughness": torch.tensor(0.7),
                "anisotropic": torch.tensor(0.5)
            },
            {
                "name": "Car Paint",
                "base_color": torch.tensor([0.1, 0.2, 0.8]),
                "metallic": torch.tensor(0.0),
                "roughness": torch.tensor(0.2),
                "specular_level": torch.tensor(0.5),
                "clearcoat": torch.tensor(1.0),
                "clearcoat_roughness": torch.tensor(0.1)
            }
        ]

        for material in materials:
            print(f"渲染材质: {material['name']}")
            renderer = PathTracer(width=500, height=500, max_depth=5)
            renderer.bsdf.set_parameters(**material)
            image = renderer.render(scene, camera)

            # 构建完整的保存路径，使用指定文件夹
            filename = f"{material['name'].replace(' ', '_')}.png"
            filepath = os.path.join(output_dir, filename)

            plt.figure(figsize=(10, 8))
            plt.imshow(image.cpu().numpy())
            plt.title(material['name'])
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"已保存到: {filepath}")


    # 运行材质测试
    test_different_materials(output_dir="C:\\Users\\steven\\Desktop\\my_renders")