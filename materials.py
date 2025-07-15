def get_materials():
    return [
        # 金属度（metallic）测试
        {
            "name": "Metallic_0.00",
            "params": {
                "base_color": [0.8, 0.6, 0.2],
                "metallic": 0.0,  # 非金属（仅修改此项）
                "roughness": 0.3,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Metallic_0.25",
            "params": {
                "base_color": [0.8, 0.6, 0.2],
                "metallic": 0.25,  # 非金属（仅修改此项）
                "roughness": 0.3,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Metallic_0.50",
            "params": {
                "base_color": [0.8, 0.6, 0.2],
                "metallic": 0.5,  # 非金属（仅修改此项）
                "roughness": 0.3,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Metallic_0.75",
            "params": {
                "base_color": [0.8, 0.6, 0.2],
                "metallic": 0.75,  # 非金属（仅修改此项）
                "roughness": 0.3,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Metallic_1.00",
            "params": {
                "base_color": [0.8, 0.6, 0.2],
                "metallic": 1.0,  # 金属（仅修改此项）
                "roughness": 0.3,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        # 粗糙度（roughness）测试
        {
            "name": "Roughness_0.00",
            "params": {
                "base_color": [0.5, 0.5, 0.5],
                "metallic": 0.0,
                "roughness": 0.0,  # 低粗糙度（高光锐利）（仅修改此项）
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Roughness_0.25",
            "params": {
                "base_color": [0.5, 0.5, 0.5],
                "metallic": 0.0,
                "roughness": 0.25,  # 低粗糙度（高光锐利）（仅修改此项）
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Roughness_0.50",
            "params": {
                "base_color": [0.5, 0.5, 0.5],
                "metallic": 0.0,
                "roughness": 0.5,  # 低粗糙度（高光锐利）（仅修改此项）
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Roughness_0.75",
            "params": {
                "base_color": [0.5, 0.5, 0.5],
                "metallic": 0.0,
                "roughness": 0.75,  # 低粗糙度（高光锐利）（仅修改此项）
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Roughness_1.00",
            "params": {
                "base_color": [0.5, 0.5, 0.5],
                "metallic": 0.0,
                "roughness": 1.0,  # 高粗糙度（高光模糊）（仅修改此项）
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        # 次表面散射（subsurface）测试
        {
            "name": "Subsurface_0.00",
            "params": {
                "base_color": [0.85, 0.65, 0.6],  # 肤色
                "metallic": 0.0,
                "roughness": 0.4,
                "specular": 0.5,
                "subsurface": 0.0,  # 无次表面散射（仅修改此项）
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Subsurface_0.25",
            "params": {
                "base_color": [0.85, 0.65, 0.6],  # 肤色
                "metallic": 0.0,
                "roughness": 0.4,
                "specular": 0.5,
                "subsurface": 0.25,  # 无次表面散射（仅修改此项）
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Subsurface_0.50",
            "params": {
                "base_color": [0.85, 0.65, 0.6],  # 肤色
                "metallic": 0.0,
                "roughness": 0.4,
                "specular": 0.5,
                "subsurface": 0.5,  # 无次表面散射（仅修改此项）
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Subsurface_0.75",
            "params": {
                "base_color": [0.85, 0.65, 0.6],  # 肤色
                "metallic": 0.0,
                "roughness": 0.4,
                "specular": 0.5,
                "subsurface": 0.75,  # 无次表面散射（仅修改此项）
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Subsurface_1.00",
            "params": {
                "base_color": [0.85, 0.65, 0.6],  # 肤色
                "metallic": 0.0,
                "roughness": 0.4,
                "specular": 0.5,
                "subsurface": 1.0,  # 强次表面散射（仅修改此项）
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        # 镜面强度（specular）测试
        {
            "name": "Specular_0.00",
            "params": {
                "base_color": [0.5, 0.5, 0.5],
                "metallic": 0.0,
                "roughness": 0.3,
                "specular": 0.0,  # 低镜面强度（仅修改此项）
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Specular_0.25",
            "params": {
                "base_color": [0.5, 0.5, 0.5],
                "metallic": 0.0,
                "roughness": 0.3,
                "specular": 0.25,  # 低镜面强度（仅修改此项）
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Specular_0.50",
            "params": {
                "base_color": [0.5, 0.5, 0.5],
                "metallic": 0.0,
                "roughness": 0.3,
                "specular": 0.5,  # 低镜面强度（仅修改此项）
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Specular_0.75",
            "params": {
                "base_color": [0.5, 0.5, 0.5],
                "metallic": 0.0,
                "roughness": 0.3,
                "specular": 0.75,  # 低镜面强度（仅修改此项）
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Specular_1.00",
            "params": {
                "base_color": [0.5, 0.5, 0.5],
                "metallic": 0.0,
                "roughness": 0.3,
                "specular": 1.0,  # 高镜面强度（仅修改此项）
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        # 镜面颜色（specular_tint）测试
        {
            "name": "SpecularTint_0.00",
            "params": {
                "base_color": [1.0, 0.2, 0.2],  # 红色
                "metallic": 0.0,
                "roughness": 0.3,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,  # 无颜色混合（仅修改此项）
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "SpecularTint_0.25",
            "params": {
                "base_color": [1.0, 0.2, 0.2],  # 红色
                "metallic": 0.0,
                "roughness": 0.3,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.25,  # 无颜色混合（仅修改此项）
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "SpecularTint_0.50",
            "params": {
                "base_color": [1.0, 0.2, 0.2],  # 红色
                "metallic": 0.0,
                "roughness": 0.3,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.50,  # 无颜色混合（仅修改此项）
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "SpecularTint_0.75",
            "params": {
                "base_color": [1.0, 0.2, 0.2],  # 红色
                "metallic": 0.0,
                "roughness": 0.3,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.75,  # 无颜色混合（仅修改此项）
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "SpecularTint_1.00",
            "params": {
                "base_color": [1.0, 0.2, 0.2],  # 红色
                "metallic": 0.0,
                "roughness": 0.3,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 1.0,  # 镜面颜色=base_color（仅修改此项）
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        # 各向异性（anisotropic）测试
        {
            "name": "Anisotropic_0.00",
            "params": {
                "base_color": [0.9, 0.9, 0.9],  # 银色
                "metallic": 1.0,
                "roughness": 0.5,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,  # 无各向异性（仅修改此项）
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Anisotropic_0.25",
            "params": {
                "base_color": [0.9, 0.9, 0.9],  # 银色
                "metallic": 1.0,
                "roughness": 0.5,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.25,  # 无各向异性（仅修改此项）
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Anisotropic_0.50",
            "params": {
                "base_color": [0.9, 0.9, 0.9],  # 银色
                "metallic": 1.0,
                "roughness": 0.5,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.5,  # 无各向异性（仅修改此项）
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Anisotropic_0.75",
            "params": {
                "base_color": [0.9, 0.9, 0.9],  # 银色
                "metallic": 1.0,
                "roughness": 0.5,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.75,  # 无各向异性（仅修改此项）
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Anisotropic_1.00",
            "params": {
                "base_color": [0.9, 0.9, 0.9],  # 银色
                "metallic": 1.0,
                "roughness": 0.5,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 1.0,  # 强各向异性（仅修改此项）
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        # 织物光泽（sheen）测试
        {
            "name": "Sheen_0.00",
            "params": {
                "base_color": [0.2, 0.2, 0.8],  # 蓝色
                "metallic": 0.0,
                "roughness": 0.7,
                "specular": 0.2,
                "subsurface": 0.0,
                "sheen": 0.0,  # 无织物光泽（仅修改此项）
                "sheen_tint": 0.5,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Sheen_0.25",
            "params": {
                "base_color": [0.2, 0.2, 0.8],  # 蓝色
                "metallic": 0.0,
                "roughness": 0.7,
                "specular": 0.2,
                "subsurface": 0.0,
                "sheen": 0.25,  # 无织物光泽（仅修改此项）
                "sheen_tint": 0.5,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Sheen_0.50",
            "params": {
                "base_color": [0.2, 0.2, 0.8],  # 蓝色
                "metallic": 0.0,
                "roughness": 0.7,
                "specular": 0.2,
                "subsurface": 0.0,
                "sheen": 0.5,  # 无织物光泽（仅修改此项）
                "sheen_tint": 0.5,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Sheen_0.75",
            "params": {
                "base_color": [0.2, 0.2, 0.8],  # 蓝色
                "metallic": 0.0,
                "roughness": 0.7,
                "specular": 0.2,
                "subsurface": 0.0,
                "sheen": 0.75,  # 无织物光泽（仅修改此项）
                "sheen_tint": 0.5,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Sheen_1.00",
            "params": {
                "base_color": [0.2, 0.2, 0.8],  # 蓝色
                "metallic": 0.0,
                "roughness": 0.7,
                "specular": 0.2,
                "subsurface": 0.0,
                "sheen": 1.0,  # 强织物光泽（仅修改此项）
                "sheen_tint": 0.5,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        # 清漆层（clearcoat）测试
        {
            "name": "Clearcoat_0.00",
            "params": {
                "base_color": [0.1, 0.2, 0.8],  # 蓝色
                "metallic": 0.0,
                "roughness": 0.2,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,  # 无清漆层（仅修改此项）
                "clearcoat_roughness": 0.1,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Clearcoat_0.25",
            "params": {
                "base_color": [0.1, 0.2, 0.8],  # 蓝色
                "metallic": 0.0,
                "roughness": 0.2,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.25,  # 无清漆层（仅修改此项）
                "clearcoat_roughness": 0.1,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Clearcoat_0.50",
            "params": {
                "base_color": [0.1, 0.2, 0.8],  # 蓝色
                "metallic": 0.0,
                "roughness": 0.2,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.50,  # 无清漆层（仅修改此项）
                "clearcoat_roughness": 0.1,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Clearcoat_0.75",
            "params": {
                "base_color": [0.1, 0.2, 0.8],  # 蓝色
                "metallic": 0.0,
                "roughness": 0.2,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.75,  # 无清漆层（仅修改此项）
                "clearcoat_roughness": 0.1,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Clearcoat_1.00",
            "params": {
                "base_color": [0.1, 0.2, 0.8],  # 蓝色
                "metallic": 0.0,
                "roughness": 0.2,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 1.0,  # 全清漆层（仅修改此项）
                "clearcoat_roughness": 0.1,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        # 透射强度（transmission）测试
        {
            "name": "Transmission_0.00",
            "params": {
                "base_color": [0.98, 0.98, 0.98],  # 近白色
                "metallic": 0.0,
                "roughness": 0.0,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.0,  # 不透明（仅修改此项）
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Transmission_0.25",
            "params": {
                "base_color": [0.98, 0.98, 0.98],  # 近白色
                "metallic": 0.0,
                "roughness": 0.0,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.25,  # 不透明（仅修改此项）
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Transmission_0.50",
            "params": {
                "base_color": [0.98, 0.98, 0.98],  # 近白色
                "metallic": 0.0,
                "roughness": 0.0,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.5,  # 不透明（仅修改此项）
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Transmission_0.75",
            "params": {
                "base_color": [0.98, 0.98, 0.98],  # 近白色
                "metallic": 0.0,
                "roughness": 0.0,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 0.75,  # 不透明（仅修改此项）
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Transmission_1.00",
            "params": {
                "base_color": [0.98, 0.98, 0.98],  # 近白色
                "metallic": 0.0,
                "roughness": 0.0,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.0,
                "ior": 1.5,
                "transmission": 1.0,  # 完全透明（仅修改此项）
                "transmission_roughness": 0.0
            }
        },

        {
            "name": "Anisotropic_Horizontal",
            "params": {
                "base_color": [0.9, 0.9, 0.9],  # 银色
                "metallic": 1.0,
                "roughness": 0.3,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.8,  # 强各向异性
                "tangent": [1.0, 0.0, 0.0],  # 水平方向（仅修改此项）
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        },
        {
            "name": "Anisotropic_Vertical",
            "params": {
                "base_color": [0.9, 0.9, 0.9],  # 银色
                "metallic": 1.0,
                "roughness": 0.3,
                "specular": 0.5,
                "subsurface": 0.0,
                "sheen": 0.0,
                "clearcoat": 0.0,
                "specular_tint": 0.0,
                "anisotropic": 0.8,  # 强各向异性
                "tangent": [0.0, 1.0, 0.0],  # 垂直方向（仅修改此项）
                "ior": 1.5,
                "transmission": 0.0,
                "transmission_roughness": 0.0
            }
        }
    ]

# materials = [
#     {
#         "name": "Red_Diffuse_Only",
#         "params": {
#             "base_color": [1.0, 0.0, 0.0],
#             "metallic": 0.0,
#             "roughness": 0.5,
#             "specular": 0.0,
#             "subsurface": 0.0,
#             "sheen": 0.0,
#             "clearcoat": 0.0,
#             "specular_tint": 0.0,
#             "anisotropic": 0.0,
#             "ior": 1.5,
#             "transmission": 0.0,
#             "transmission_roughness": 0.0
#         }
#     },
#     {
#         "name": "Metallic_Gold",
#         "params": {
#             "base_color": [0.8, 0.6, 0.2],  # 金色
#             "metallic": 1.0,  # 纯金属
#             "roughness": 0.3,  # 中等粗糙度
#             "specular": 0.5,
#             "subsurface": 0.0,
#             "sheen": 0.0,
#             "clearcoat": 0.0,
#             "specular_tint": 0.0,
#             "anisotropic": 0.0,
#             "ior": 1.5,
#             "transmission": 0.0,
#             "transmission_roughness": 0.0
#         }
#     },
#     {
#         "name": "Rough_Matte",
#         "params": {
#             "base_color": [0.2, 0.6, 0.8],  # 蓝色
#             "metallic": 0.0,
#             "roughness": 0.9,  # 高粗糙度（接近哑光）
#             "specular": 0.5,
#             "subsurface": 0.0,
#             "sheen": 0.0,
#             "clearcoat": 0.0,
#             "specular_tint": 0.0,
#             "anisotropic": 0.0,
#             "ior": 1.5,
#             "transmission": 0.0,
#             "transmission_roughness": 0.0
#         }
#     },
#     {
#         "name": "Smooth_Mirror",
#         "params": {
#             "base_color": [0.9, 0.9, 0.9],  # 灰色
#             "metallic": 1.0,
#             "roughness": 0.01,  # 接近镜面
#             "specular": 0.5,
#             "subsurface": 0.0,
#             "sheen": 0.0,
#             "clearcoat": 0.0,
#             "specular_tint": 0.0,
#             "anisotropic": 0.0,
#             "ior": 1.5,
#             "transmission": 0.0,
#             "transmission_roughness": 0.0
#         }
#     },
#     {
#         "name": "Subsurface_Skin",
#         "params": {
#             "base_color": [0.85, 0.65, 0.6],  # 肤色
#             "metallic": 0.0,
#             "roughness": 0.4,
#             "specular": 0.5,
#             "subsurface": 0.7,  # 高次表面散射
#             "sheen": 0.0,
#             "clearcoat": 0.0,
#             "specular_tint": 0.0,
#             "anisotropic": 0.0,
#             "ior": 1.5,
#             "transmission": 0.0,
#             "transmission_roughness": 0.0
#         }
#     },
#     {
#         "name": "Sheen_Cloth",
#         "params": {
#             "base_color": [0.2, 0.2, 0.8],  # 深蓝色
#             "metallic": 0.0,
#             "roughness": 0.7,
#             "specular": 0.2,
#             "subsurface": 0.0,
#             "sheen": 0.8,  # 高边缘光泽
#             "sheen_tint": 0.5,
#             "clearcoat": 0.0,
#             "specular_tint": 0.0,
#             "anisotropic": 0.0,
#             "ior": 1.5,
#             "transmission": 0.0,
#             "transmission_roughness": 0.0
#         }
#     },
#     {
#         "name": "Clearcoat_CarPaint",
#         "params": {
#             "base_color": [0.1, 0.2, 0.8],  # 深蓝色
#             "metallic": 0.0,
#             "roughness": 0.2,
#             "specular": 0.5,
#             "subsurface": 0.0,
#             "sheen": 0.0,
#             "clearcoat": 1.0,  # 完全清漆层
#             "clearcoat_roughness": 0.1,  # 清漆层粗糙度
#             "specular_tint": 0.0,
#             "anisotropic": 0.0,
#             "ior": 1.5,
#             "transmission": 0.0,
#             "transmission_roughness": 0.0
#         }
#     },
#     {
#         "name": "Anisotropic_BrushedMetal",
#         "params": {
#             "base_color": [0.9, 0.9, 0.92],  # 银色
#             "metallic": 1.0,
#             "roughness": 0.5,
#             "specular": 0.5,
#             "subsurface": 0.0,
#             "sheen": 0.0,
#             "clearcoat": 0.0,
#             "specular_tint": 0.0,
#             "anisotropic": 0.8,  # 高各向异性（拉丝效果）
#             "ior": 1.5,
#             "transmission": 0.0,
#             "transmission_roughness": 0.0
#         }
#     },
#     {
#         "name": "Glass",
#         "params": {
#             "base_color": [0.98, 0.98, 0.98],  # 近白色
#             "metallic": 0.0,
#             "roughness": 0.0,
#             "specular": 0.5,
#             "subsurface": 0.0,
#             "sheen": 0.0,
#             "clearcoat": 0.0,
#             "specular_tint": 0.0,
#             "anisotropic": 0.0,
#             "ior": 1.5,  # 玻璃折射率
#             "transmission": 1.0,  # 完全透明
#             "transmission_roughness": 0.0  # 完全光滑
#         }
#     },
#     {
#         "name": "Translucent_Perlin",
#         "params": {
#             "base_color": [0.2, 0.8, 0.4],  # 绿色
#             "metallic": 0.0,
#             "roughness": 0.3,
#             "specular": 0.5,
#             "subsurface": 0.0,
#             "sheen": 0.0,
#             "clearcoat": 0.0,
#             "specular_tint": 0.0,
#             "anisotropic": 0.0,
#             "ior": 1.3,  # 水/塑料折射率
#             "transmission": 0.7,  # 部分透明
#             "transmission_roughness": 0.2  # 半透明磨砂效果
#         }
#     }
# ]
