# CGLab_BSDF
Disney Principled BSDF 渲染及查看器
项目概述
本项目实现了一个基于迪士尼原则性 BSDF（Bidirectional Scattering Distribution Function） 的路径追踪渲染器，旨在通过物理精确的光线模拟，生成具有真实感的材质效果。系统支持多种材质参数调整（如金属度、粗糙度、各向异性等），并通过多进程并行渲染提升效率。同时本项目提供了一个渲染材质球查看器，方便实时调整和预览。
核心功能包括：
- 实现迪士尼原则性 BSDF 的完整组件（漫反射、镜面反射、清漆层、织物光泽、透射等）
- 支持复杂材质参数调整，覆盖金属、塑料、布料、玻璃等多种材质类型
- 多进程并行渲染，提升大分辨率图像的渲染效率
- 可视化 UI 界面，支持实时调整参数并预览渲染结果
项目架构
项目目录结构

├── CG_Disney_Principled_BSDF.py  # 核心渲染实现（BSDF、路径追踪器等）

├── materials.py                  # 材质参数预设（金属、塑料、玻璃等）

└── viewer.py                     # 渲染材质球查看器UI界面

核心类说明
1. DisneyPrincipledBSDF
实现迪士尼原则性 BSDF 模型的核心类，包含多种材质组件：
  - 漫反射项（diffuse_term）：模拟非金属材质的漫反射效果
  - 镜面反射项（specular_term）：处理高光反射，支持各向异性
  - 清漆层（clearcoat_term）：模拟透明涂层（如汽车漆）的额外高光
  - 织物光泽（sheen_term）：模拟天鹅绒等织物的边缘发光效果
  - 透射项（transmission_term）：模拟玻璃等透明材质的折射效果
2. PathTracer
路径追踪渲染器类，负责：
  - 生成相机射线（通过 Camera 类）
  - 递归追踪光线与场景的交互（trace_ray）
  - 多进程并行渲染（利用 multiprocessing 提高效率）
3. 辅助类
  - Ray：表示光线（起点和方向）
  - SphereScene：简单场景类，支持射线与球体的相交检测
  - Camera：相机模型，负责从像素坐标生成射线方向
  - BSDFRendererUI：UI 界面类，提供参数调整和渲染预览
使用方式
1. 环境准备
安装依赖库：
pip install torch matplotlib pillow tkinter
2. 基本使用流程
方式一：通过 UI 界面交互
运行材质球查看器，调整参数并渲染：
python viewer.py或运行viewer.exe
- 界面包含参数滑块（金属度、粗糙度等）、颜色选择器
- 支持实时预览渲染结果，调整后点击 "渲染" 按钮更新图像
方式二：脚本方式渲染
在代码materials.py中配置材质参数并渲染：
1. python
2. 运行
python CG_Disney_Principled_BSDF.py
参数说明
通过 UI 界面或set_parameters方法可调整的核心参数：
- base_color：基础颜色（RGB 值，如 [1,0,0] 为红色）
- metallic：金属度（0 = 非金属，1 = 金属）
- roughness：粗糙度（0 = 镜面，1 = 哑光）
- anisotropic：各向异性（0 = 各向同性，±1 = 水平 / 垂直拉伸高光）
- sheen：织物光泽强度（0 = 无，1 = 强）
- clearcoat：清漆层强度（0 = 无，1 = 强）
- transmission：透射率（0 = 不透明，1 = 完全透明，用于玻璃等）
