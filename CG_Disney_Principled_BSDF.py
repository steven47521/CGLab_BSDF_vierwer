# -*- coding: utf-8 -*-
import torch
import math
import matplotlib.pyplot as plt
import os
import multiprocessing
from multiprocessing import Pool

# 一些物理常量与常用函数
PI = torch.tensor(math.pi)
EPS = 1e-8


def safe_divide(a, b):
    """安全除法，避免除以0"""
    return a / (b + EPS)


def schlick_fresnel(cos_theta, f0):
    """Schlick 菲涅尔近似"""
    return f0 + (1.0 - f0) * torch.pow(1.0 - cos_theta, 5.0)


def trowbridge_reitz_distribution(n_dot_h, alpha_x, alpha_y):
    """
    Trowbridge-Reitz（GGX）微表面分布函数
    n_dot_h: 法线与半程向量的点积
    alpha_x: x 方向粗糙度相关参数
    alpha_y: y 方向粗糙度相关参数
    """
    alpha_x_sq = alpha_x * alpha_x
    alpha_y_sq = alpha_y * alpha_y
    denom = (n_dot_h * n_dot_h) * (alpha_x_sq * alpha_y_sq - 1.0) + 1.0
    return (alpha_x * alpha_y) / (PI * denom * denom)


def smith_ggx_geometry(n_dot_v, n_dot_l, alpha):
    """Smith 几何函数（GGX 形式）"""
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
        # 材质参数
        self.base_color = torch.tensor([0.0, 0.0, 0.0])
        self.subsurface = torch.tensor(0.5)
        self.metallic = torch.tensor(0.5)
        self.specular = torch.tensor(0.1)
        self.specular_tint = torch.tensor(0.1)
        self.roughness = torch.tensor(0.5)
        self.anisotropic = torch.tensor(0.5)
        self.sheen = torch.tensor(0.5)
        self.sheen_tint = torch.tensor(0.5)
        self.clearcoat = torch.tensor(0.5)
        self.clearcoat_roughness = torch.tensor(0.5)
        self.ior = torch.tensor(1.5)
        self.transmission = torch.tensor(0.5)
        self.transmission_roughness = torch.tensor(0.5)
        self.tangent = torch.tensor([1.0, 0.0, 0.0])

    def set_parameters(self, **kwargs):
        """动态设置材质参数"""
        if "tangent" in kwargs:
            self.tangent = torch.tensor(kwargs["tangent"], dtype=torch.float32)
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, (int, float)):
                    value = torch.tensor(value, dtype=torch.float32)
                elif isinstance(value, list):
                    value = torch.tensor(value, dtype=torch.float32)
                setattr(self, key, value)

    def diffuse_term(self, n_dot_l, n_dot_v, v_dot_h):
        """漫反射项（含次表面散射）"""
        metallic = self.metallic.to(torch.float32)
        if metallic > 0.99:  # 金属无漫反射
            return torch.zeros_like(self.base_color)

        # 菲涅尔项修正
        fl = schlick_fresnel(n_dot_l, torch.tensor(0.0))
        fv = schlick_fresnel(n_dot_v, torch.tensor(0.0))

        # 能量守恒修正
        energy_bias = 0.5
        fd90 = energy_bias + 2.0 * v_dot_h ** 2 * self.roughness
        fd_l = 1.0 + (fd90 - 1.0) * fl
        fd_v = 1.0 + (fd90 - 1.0) * fv

        # 次表面散射近似
        subsurface = self.subsurface
        if subsurface > 0:
            ss = 1.25 * (
                    (1.0 - (fl * fv) / 2.0) -
                    (1.0 - fl) * (1.0 - fv) * (1.0 / 3.0)
            )
            subsurface_term = subsurface * ss / PI
        else:
            subsurface_term = 0.0

        # 基础漫反射
        diffuse_base = (1.0 - subsurface) * fd_l * fd_v / PI

        # 总漫反射
        total_diffuse = self.base_color * (1.0 - metallic) * (diffuse_base + subsurface_term)
        return torch.clamp(total_diffuse, 0.0, 1.0)

    def specular_term(self, n, v, l):
        """镜面反射项"""

        n = n / torch.norm(n, dim=-1, keepdim=True).clamp_min(EPS)
        v = v / torch.norm(v, dim=-1, keepdim=True).clamp_min(EPS)
        l = l / torch.norm(l, dim=-1, keepdim=True).clamp_min(EPS)

        # 计算半程向量
        h = (v + l) / torch.norm(v + l, dim=-1, keepdim=True).clamp_min(EPS)

        # 计算点积
        n_dot_h = torch.clamp(torch.sum(n * h, dim=-1, keepdim=True), 0.0, 1.0)
        n_dot_v = torch.clamp(torch.sum(n * v, dim=-1, keepdim=True), 0.0, 1.0)
        n_dot_l = torch.clamp(torch.sum(n * l, dim=-1, keepdim=True), 0.0, 1.0)
        v_dot_h = torch.clamp(torch.sum(v * h, dim=-1, keepdim=True), 0.0, 1.0)

        # 金属/非金属反射率基础值（f0）
        dielectric_f0 = 0.04 * self.specular  # 非金属基础反射率
        if self.specular_tint > 0:
            dielectric_f0 = torch.lerp(
                torch.tensor(dielectric_f0, dtype=torch.float32),
                self.base_color * dielectric_f0,
                self.specular_tint
            )
        metallic_f0 = self.base_color  # 金属反射率=基础颜色
        f0 = torch.lerp(dielectric_f0, metallic_f0, self.metallic)

        # 菲涅尔项
        F = schlick_fresnel(v_dot_h, f0)

        # 各向异性处理
        # 确保切线向量与法线正交（Gram-Schmidt正交化）
        tangent = self.tangent - torch.sum(self.tangent * n, dim=-1, keepdim=True) * n
        tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp_min(EPS)

        # 计算副法线（与法线和切线正交）
        binormal = torch.cross(n, tangent, dim=-1)

        # 计算h在切线空间的投影
        h_t = torch.sum(h * tangent, dim=-1, keepdim=True)
        h_b = torch.sum(h * binormal, dim=-1, keepdim=True)

        # 各向异性GGX分布
        alpha_x, alpha_y = compute_anisotropy_alphas(self.roughness, self.anisotropic)

        # 各向异性GGX分布函数
        alpha_x_sq = alpha_x * alpha_x
        alpha_y_sq = alpha_y * alpha_y
        term = (h_t * h_t) / alpha_x_sq + (h_b * h_b) / alpha_y_sq + (n_dot_h * n_dot_h)
        D = 1.0 / (PI * alpha_x * alpha_y * term * term)

        # 几何项（Smith-GGX，考虑各向异性）
        lambda_v = n_dot_v * torch.sqrt(
            (h_t * h_t * alpha_x_sq + h_b * h_b * alpha_y_sq) / (n_dot_h * n_dot_h)
        )
        lambda_l = n_dot_l * torch.sqrt(
            (h_t * h_t * alpha_x_sq + h_b * h_b * alpha_y_sq) / (n_dot_h * n_dot_h)
        )
        G = 1.0 / (1.0 + lambda_v + lambda_l)

        # 镜面BRDF公式：(D * G * F) / (4 * n·l * n·v)
        denom = 4.0 * n_dot_l * n_dot_v
        specular_brdf = safe_divide(D * G * F, denom)

        return torch.clamp(specular_brdf, 0.0, 10.0)

    def clearcoat_term(self, n, v, l):
        """清漆层项"""
        if self.clearcoat <= 0:
            return torch.zeros_like(n)

        # 清漆层参数（固定折射率1.5，粗糙度单独控制）
        h = (v + l) / torch.norm(v + l, dim=-1, keepdim=True).clamp_min(EPS)
        n_dot_h = torch.clamp(torch.sum(n * h, dim=-1, keepdim=True), 0.0, 1.0)
        n_dot_v = torch.clamp(torch.sum(n * v, dim=-1, keepdim=True), 0.0, 1.0)
        n_dot_l = torch.clamp(torch.sum(n * l, dim=-1, keepdim=True), 0.0, 1.0)

        f0_clearcoat = torch.tensor(0.04, dtype=torch.float32)
        F = schlick_fresnel(torch.max(n_dot_h, torch.tensor(0.0)), f0_clearcoat)

        # 清漆层分布（粗糙度单独控制）
        alpha_clearcoat = self.clearcoat_roughness ** 2 * 0.15
        D_clearcoat = trowbridge_reitz_distribution(n_dot_h, alpha_clearcoat, alpha_clearcoat)

        # 清漆层几何项
        G_clearcoat = 0.5 / (n_dot_l + n_dot_v - n_dot_l * n_dot_v)

        # 清漆层 BRDF
        clearcoat_brdf = 1.25 * (D_clearcoat * G_clearcoat * F)
        return self.clearcoat * clearcoat_brdf

    def sheen_term(self, n, v, l):
        """织物光泽项"""
        if self.sheen <= 0 or self.metallic > 0.99:
            return torch.zeros_like(n)

        h = (v + l) / torch.norm(v + l, dim=-1, keepdim=True).clamp_min(EPS)
        n_dot_h = torch.clamp(torch.sum(n * h, dim=-1, keepdim=True), 0.0, 1.0)

        # 织物颜色混合
        sheen_color = torch.lerp(
            torch.tensor(1.0, dtype=torch.float32),
            self.base_color,
            self.sheen_tint
        )

        # 织物光泽公式（基于n·h的平滑过渡）
        sheen_brdf = sheen_color * (1.0 - torch.pow(n_dot_h, 5.0))
        return self.sheen * sheen_brdf

    def transmission_term(self, n, v, l):
        """透射项（透明/半透明材质）"""
        if self.transmission <= 0 or self.metallic > 0.99:
            return torch.zeros_like(n)

        # 计算折射方向（基于斯涅尔定律）
        eta = self.ior  # 折射率
        n_dot_v = torch.sum(n * v, dim=-1, keepdim=True).clamp(-1.0, 1.0)
        n_dot_l = torch.sum(n * l, dim=-1, keepdim=True).clamp(-1.0, 1.0)

        # 确定光线方向（入射/折射）
        entering = n_dot_v > 0
        eta_i = torch.where(entering, torch.tensor(1.0), eta)
        eta_t = torch.where(entering, eta, torch.tensor(1.0))
        n = torch.where(entering.unsqueeze(-1), n, -n)

        # 斯涅尔定律计算折射角
        sin_theta_i = torch.sqrt(torch.clamp(1.0 - n_dot_v ** 2, 0.0, 1.0))
        sin_theta_t = (eta_i / eta_t) * sin_theta_i

        # 全反射检查
        if (sin_theta_t ** 2 > 1.0).any():
            return torch.zeros_like(n)

        cos_theta_t = torch.sqrt(torch.clamp(1.0 - sin_theta_t ** 2, 0.0, 1.0))

        # 透射方向计算
        t = (eta_i / eta_t) * (-v) + (
                (eta_i / eta_t) * n_dot_v - cos_theta_t
        ) * n

        # 磨砂透射（粗糙度影响）
        if self.transmission_roughness > 0:
            alpha_tr = self.transmission_roughness ** 2
            D_tr = trowbridge_reitz_distribution(torch.sum(n * t, dim=-1, keepdim=True), alpha_tr, alpha_tr)
        else:
            D_tr = 1.0

        # 透射菲涅尔项（能量守恒）
        F = schlick_fresnel(torch.abs(n_dot_v), torch.tensor(0.04))
        T = 1.0 - F

        # 透射 BRDF 公式
        transmission_brdf = (self.base_color * T * D_tr) / (eta_t ** 2)
        return self.transmission * transmission_brdf

    def forward(self, n, v, l):
        """总 BSDF 计算：所有项求和"""

        n = n / torch.norm(n, dim=-1, keepdim=True).clamp_min(EPS)
        v = v / torch.norm(v, dim=-1, keepdim=True).clamp_min(EPS)
        l = l / torch.norm(l, dim=-1, keepdim=True).clamp_min(EPS)

        # 计算各分量
        diffuse = self.diffuse_term(
            torch.sum(n * l, dim=-1, keepdim=True),
            torch.sum(n * v, dim=-1, keepdim=True),
            torch.sum(v * ((v + l) / torch.norm(v + l, dim=-1, keepdim=True).clamp_min(EPS)), dim=-1, keepdim=True)
        )
        specular = self.specular_term(n, v, l)
        clearcoat = self.clearcoat_term(n, v, l)
        sheen = self.sheen_term(n, v, l)
        transmission = self.transmission_term(n, v, l)

        # 确保能量守恒
        if self.metallic > 0.99:
            diffuse = torch.zeros_like(diffuse)
            transmission = torch.zeros_like(transmission)

        # 总贡献
        total = diffuse + specular + clearcoat + sheen + transmission
        return torch.clamp(total, 0.0, 10.0)


class PathTracer:
    def __init__(self, width=400, height=400, max_depth=5):
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.bsdf = DisneyPrincipledBSDF()

    def render(self, scene, camera, num_processes=None):
        """使用多进程并行渲染"""
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()

            # 分割渲染任务为多个块
        rows_per_process = self.height // num_processes
        row_blocks = [(i * rows_per_process, min((i + 1) * rows_per_process, self.height))
                      for i in range(num_processes)]

        # 准备进程池参数
        pool_args = [(camera, scene, start_row, end_row) for start_row, end_row in row_blocks]

        # 使用进程池并行处理
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(self.render_row_block, pool_args)

        # 合并结果
        image = torch.zeros((self.height, self.width, 3))
        for start_row, end_row, block in results:
            image[start_row:end_row] = block

        return image

    def render_row_block(self, camera, scene, start_row, end_row):
        """渲染指定行范围的像素块"""
        block_height = end_row - start_row
        block = torch.zeros((block_height, self.width, 3))

        for y in range(start_row, end_row):
            for x in range(self.width):
                ray = camera.generate_ray(x, y, self.width, self.height)
                color = self.trace_ray(ray, scene, self.max_depth)
                block[y - start_row, x] = color

        return start_row, end_row, block

    def trace_ray(self, ray, scene, depth):
        """递归追踪射线"""
        if depth == 0:
            return torch.zeros(3)

        hit, n, p = scene.intersect_sphere(ray)
        if not hit:
            return torch.zeros(3)  # 背景为黑色

        # 计算光线方向
        l = torch.tensor([3, 5.0, -13])  # 设置光源方向
        l = l / torch.norm(l)
        v = -ray.direction  # 视线方向是射线方向的反方向

        # 获取 BSDF 值
        bsdf_value = self.bsdf(n.unsqueeze(0), v.unsqueeze(0), l.unsqueeze(0))[0]

        # 直接光照贡献
        light_intensity = torch.tensor([1.0, 1.0, 1.0])  # 白色光源
        direct_color = bsdf_value * light_intensity

        # 间接光照
        indirect_color = torch.zeros(3)

        return direct_color + indirect_color


# 辅助的射线和球体场景类
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
        self.eye = eye  # 相机位置
        self.lookat = lookat  # 目标点
        self.up = up  # 上方向量

        # 1. 计算相机坐标系的三个轴（forward, right, up）
        self.forward = (lookat - eye) / torch.norm(lookat - eye)  # 前向向量
        self.right = torch.cross(self.forward, up, dim=0)  # 右方向量
        self.right = self.right / torch.norm(self.right)  # 归一化
        self.up = torch.cross(self.right, self.forward, dim=0)  # 修正上方向量（与前向、右方垂直）
        self.up = self.up / torch.norm(self.up)  # 归一化

        # 2. 计算fov_scale
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
    sphere_center = torch.tensor([0.0, 0.0, 0.5])
    sphere_radius = torch.tensor(0.5)
    scene = SphereScene(sphere_center, sphere_radius)


    # 测试不同材质
    def test_different_materials(output_dir=None):
        from materials import get_materials
        materials = get_materials()

        for material in materials:
            print(f"渲染材质: {material['name']}")
            renderer = PathTracer(width=500, height=500, max_depth=5)
            renderer.bsdf.set_parameters(**material["params"])  # 从params中提取参数
            image = renderer.render(scene, camera)
            # 构建完整的保存路径，使用指定文件夹
            filename = f"{material['name'].replace(' ', '_')}.png"
            filepath = os.path.join(output_dir, filename)

            # 在 test_different_materials 中保存图像前
            image = torch.clamp(image, 0.0, 1.0)  # 限制颜色范围
            image = torch.pow(image, 1.0 / 2.2)  # 伽马校正
            plt.imshow(image.cpu().numpy())

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


# 材质球渲染器调用的接口函数
import matplotlib
matplotlib.use('Agg')  # 关键：禁用交互式窗口，仅后台渲染
import matplotlib.pyplot as plt

def render_material_to_path(material_params, output_dir):
    # 确保函数内也使用非交互式后端
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    plt.ioff()

    # 设置相机
    eye = torch.tensor([0.0, 0.0, -2.0])
    lookat = torch.tensor([0.0, 0.0, 0.0])
    up = torch.tensor([0.0, 1.0, 0.0])
    camera = Camera(eye, lookat, up, fov=45)

    # 设置场景（一个球体）
    sphere_center = torch.tensor([0.0, 0.0, 0.5])
    sphere_radius = torch.tensor(0.5)
    scene = SphereScene(sphere_center, sphere_radius)

    print(f"渲染材质: {material_params['name']}")
    renderer = PathTracer(width=500, height=500, max_depth=5)
    renderer.bsdf.set_parameters(**material_params["params"])  # 从params中提取参数
    image = renderer.render(scene, camera)
    # 构建完整的保存路径，使用指定文件夹
    filename = f"{material_params['name'].replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)

    # 在 test_different_materials 中保存图像前
    image = torch.clamp(image, 0.0, 1.0)  # 限制颜色范围
    image = torch.pow(image, 1.0 / 2.2)  # 伽马校正
    img_np = image.cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(img_np)
    plt.title(material_params['name'])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close('all')  # 关闭所有图像，强制释放资源（比plt.close()更彻底）

    return filepath
