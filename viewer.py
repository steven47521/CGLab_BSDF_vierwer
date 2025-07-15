import matplotlib
matplotlib.use('Agg', force=True)  # 强制使用非交互式后端（无窗口）
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互模式

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import os
from PIL import Image, ImageTk

from CG_Disney_Principled_BSDF import (
    render_material_to_path
)


class BSDFRendererUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BSDF材质球渲染器")
        self.root.geometry("1000x800")

        # 初始化材质参数
        self.material_name = tk.StringVar(value="Custom_Material")
        self.params = {
            "base_color": [0.5, 0.5, 0.5],
            "metallic": 0.0,
            "roughness": 0.5,
            "specular": 0.5,
            "subsurface": 0.0,
            "sheen": 0.0,
            "clearcoat": 0.0,
            "specular_tint": 0.0,
            "anisotropic": 0.0,
            "ior": 1.5,
            "transmission": 0.0,
            "transmission_roughness": 0.0,
        }

        # 输出目录
        self.output_dir = tk.StringVar(value=os.path.join(os.getcwd(), "renders"))

        # 创建UI组件
        self.create_widgets()

    def create_widgets(self):
        param_frame = ttk.LabelFrame(self.root, text="材质参数")
        param_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(param_frame, text="材质名称:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        name_entry = ttk.Entry(param_frame, textvariable=self.material_name, width=20)
        name_entry.grid(row=0, column=1, padx=5, pady=5)

        # 基础颜色选择
        ttk.Label(param_frame, text="基础颜色:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        color_frame = ttk.Frame(param_frame, width=200, height=30, relief=tk.SUNKEN, borderwidth=1)
        color_frame.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        self.color_label = ttk.Label(color_frame, background=self.rgb_to_hex(self.params["base_color"]))
        self.color_label.pack(fill=tk.BOTH, expand=True)
        ttk.Button(param_frame, text="选择颜色", command=self.choose_color).grid(row=1, column=2, padx=5, pady=5)

        # 参数滑块（0-1范围）
        slider_params = [
            ("金属度 (metallic)", "metallic"),
            ("粗糙度 (roughness)", "roughness"),
            ("高光 (specular)", "specular"),
            ("次表面散射 (subsurface)", "subsurface"),
            ("光泽 (sheen)", "sheen"),
            ("清漆 (clearcoat)", "clearcoat"),
            ("高光染色 (specular_tint)", "specular_tint"),
            ("各向异性 (anisotropic)", "anisotropic"),
            ("透明度 (transmission)", "transmission"),
            ("透明粗糙度 (transmission_roughness)", "transmission_roughness")
        ]

        # 折射率（特殊范围1.0-2.0）
        ttk.Label(param_frame, text="折射率 (ior):").grid(row=len(slider_params) + 2, column=0, sticky=tk.W, padx=5,
                                                          pady=5)
        ior_scale = ttk.Scale(param_frame, from_=1.0, to=2.0, value=self.params["ior"],
                              command=lambda v: self.update_param("ior", float(v)))
        ior_scale.grid(row=len(slider_params) + 2, column=1, padx=5, pady=5)
        self.ior_label = ttk.Label(param_frame, text=f"{self.params['ior']:.2f}", width=6)
        self.ior_label.grid(row=len(slider_params) + 2, column=2, padx=5, pady=5)

        # 创建滑块
        for i, (label_text, param_name) in enumerate(slider_params, start=2):
            ttk.Label(param_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            scale = ttk.Scale(param_frame, from_=0.0, to=1.0, value=self.params[param_name],
                              command=lambda v, p=param_name: self.update_param(p, float(v)))
            scale.grid(row=i, column=1, padx=5, pady=5)
            setattr(self, f"{param_name}_label", ttk.Label(param_frame, text=f"{self.params[param_name]:.2f}", width=6))
            getattr(self, f"{param_name}_label").grid(row=i, column=2, padx=5, pady=5)

        # 输出目录设置
        ttk.Label(param_frame, text="输出目录:").grid(row=len(slider_params) + 3, column=0, sticky=tk.W, padx=5,
                                                      pady=10)
        ttk.Entry(param_frame, textvariable=self.output_dir, width=25).grid(row=len(slider_params) + 3, column=1,
                                                                            padx=5, pady=10)
        ttk.Button(param_frame, text="浏览...", command=self.choose_output_dir).grid(row=len(slider_params) + 3,
                                                                                     column=2, padx=5, pady=10)

        # 渲染按钮
        render_btn = ttk.Button(param_frame, text="渲染材质球", command=self.render_material)
        render_btn.grid(row=len(slider_params) + 4, column=0, columnspan=3, pady=20)

        # 右侧预览面板
        preview_frame = ttk.LabelFrame(self.root, text="渲染预览")
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(fill=tk.NONE, expand=False)  # 不拉伸标签，保持原始尺寸

        self.preview_label.config(text="请点击渲染按钮生成预览")

    def rgb_to_hex(self, rgb):
        """将RGB值（0-1）转换为十六进制颜色"""
        r, g, b = [int(x * 255) for x in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"

    def choose_color(self):
        """选择基础颜色"""
        current_color = self.rgb_to_hex(self.params["base_color"])
        color = colorchooser.askcolor(title="选择基础颜色", initialcolor=current_color)[0]
        if color:
            rgb = [x / 255.0 for x in color]
            self.params["base_color"] = rgb
            self.color_label.config(background=self.rgb_to_hex(rgb))

    def update_param(self, param_name, value):
        """更新参数值并刷新显示"""
        self.params[param_name] = round(value, 3)
        if param_name == "ior":
            self.ior_label.config(text=f"{value:.2f}")
        else:
            getattr(self, f"{param_name}_label").config(text=f"{value:.2f}")

    def choose_output_dir(self):
        """选择输出目录"""
        dir_path = filedialog.askdirectory(title="选择渲染输出目录")
        if dir_path:
            self.output_dir.set(dir_path)

    def render_material(self):
        # 创建输出目录
        os.makedirs(self.output_dir.get(), exist_ok=True)

        # 准备材质参数字典
        material_data = {
            "name": self.material_name.get(),
            "params": self.params.copy()
        }

        try:
            # 调用渲染函数
            image_path = render_material_to_path(material_data, self.output_dir.get())

            # 显示渲染结果
            self.show_render_result(image_path)
            messagebox.showinfo("成功", f"渲染完成！\n文件保存至：{image_path}")

        except Exception as e:
            messagebox.showerror("错误", f"渲染失败：{str(e)}")

    def show_render_result(self, image_path):
        try:
            # 打开原始图片
            img = Image.open(image_path)

            # 定义预览的最大尺寸
            max_preview_size = (500, 500)

            # 计算按比例缩小后的尺寸
            img.thumbnail(max_preview_size, Image.LANCZOS)
            # 转换为Tkinter可显示的格式（保持原始尺寸）
            photo = ImageTk.PhotoImage(img)
            # 更新标签显示图片，清空提示文本
            self.preview_label.config(image=photo, text="")
            # 保留图片引用，防止被垃圾回收
            self.preview_label.image = photo
        except Exception as e:
            self.preview_label.config(text=f"无法显示图片：{str(e)}", image="")


if __name__ == "__main__":
        # 关键：告诉多进程这是主程序，子进程不要执行UI代码
        import multiprocessing
        multiprocessing.freeze_support()  # 必须放在最前面！

        # 强制设置多进程启动方式为 spawn（Windows 推荐）
        multiprocessing.set_start_method('spawn', force=True)

        # 正常启动UI
        root = tk.Tk()
        app = BSDFRendererUI(root)
        root.mainloop()