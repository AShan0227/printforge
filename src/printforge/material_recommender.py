"""
Material Recommender for 3D Print Forge

根据3D模型特征推荐最适合的打印材料。
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Material:
    """材料数据类"""
    name: str
    temperature_nozzle: int  # 喷嘴温度 (°C)
    temperature_bed: int  # 热床温度 (°C)
    bed_needed: bool  # 是否需要热床
    enclosure_needed: bool  # 是否需要封闭箱体
    tensile_strength: str  # 抗拉强度: low/medium/high
    heat_resistance: str  # 耐热性: low/medium/high
    flexibility: str  # 柔韧性: low/medium/high
    precision: str  # 精度: low/medium/high
    description: str
    suitable_for: List[str]


@dataclass
class MaterialRecommendation:
    """材料推荐结果"""
    recommended_material: Material
    alternative_materials: List[Material]
    warnings: List[str]
    reasoning: List[str]


# 内置材料数据库
MATERIAL_DATABASE = {
    "PLA": Material(
        name="PLA",
        temperature_nozzle=200,
        temperature_bed=60,
        bed_needed=True,
        enclosure_needed=False,
        tensile_strength="medium",
        heat_resistance="low",
        flexibility="low",
        precision="high",
        description="最常用的材料，打印温度低，精度高，适合初学者",
        suitable_for=["模型", "玩具", "原型", "装饰件"]
    ),
    "PETG": Material(
        name="PETG",
        temperature_nozzle=240,
        temperature_bed=80,
        bed_needed=True,
        enclosure_needed=False,
        tensile_strength="high",
        heat_resistance="medium",
        flexibility="medium",
        precision="medium",
        description="强度好，耐温，适合功能件和户外使用",
        suitable_for=["功能件", "机械零件", "户外用品", "容器"]
    ),
    "ABS": Material(
        name="ABS",
        temperature_nozzle=250,
        temperature_bed=100,
        bed_needed=True,
        enclosure_needed=True,
        tensile_strength="high",
        heat_resistance="high",
        flexibility="medium",
        precision="medium",
        description="耐冲击，耐高温，但需要封闭箱体，有翘曲风险",
        suitable_for=["汽车零件", "工具", "耐热部件", "工业件"]
    ),
    "TPU": Material(
        name="TPU",
        temperature_nozzle=220,
        temperature_bed=50,
        bed_needed=False,
        enclosure_needed=False,
        tensile_strength="medium",
        heat_resistance="medium",
        flexibility="high",
        precision="low",
        description="柔性材料，适合弹性件和密封件",
        suitable_for=["鞋垫", "软管", "密封圈", "手机壳"]
    ),
    "Resin": Material(
        name="Resin",
        temperature_nozzle=0,  # 不适用
        temperature_bed=0,
        bed_needed=False,
        enclosure_needed=True,  # 需要通风
        tensile_strength="medium",
        heat_resistance="medium",
        flexibility="low",
        precision="highest",
        description="光固化树脂，最高精度，适合小细节手办和精密零件",
        suitable_for=["手办", "珠宝", "牙科模型", "精密零件", "迷你模型"]
    ),
}


class MeshAnalyzer:
    """网格分析器"""
    
    def get_face_count(self, mesh) -> int:
        """获取面数"""
        if hasattr(mesh, 'faces'):
            return len(mesh.faces)
        elif hasattr(mesh, 'n_faces'):
            return mesh.n_faces
        elif isinstance(mesh, dict):
            return mesh.get('face_count', mesh.get('n_faces', 0))
        else:
            return getattr(mesh, 'face_count', 0)
    
    def get_max_overhang_angle(self, mesh) -> float:
        """获取最大悬垂角度（0-90度）"""
        if isinstance(mesh, dict):
            return mesh.get('max_overhang_angle', 0)
        return getattr(mesh, 'max_overhang_angle', 0)
    
    def get_wall_thickness(self, mesh) -> float:
        """获取壁厚（mm）"""
        if isinstance(mesh, dict):
            return mesh.get('wall_thickness', 2.0)
        return getattr(mesh, 'wall_thickness', 2.0)
    
    def is_flexible_model(self, mesh) -> bool:
        """判断是否为柔性模型"""
        if isinstance(mesh, dict):
            return mesh.get('is_flexible', False)
        return getattr(mesh, 'is_flexible', False)
    
    def is_high_detail(self, mesh) -> bool:
        """判断是否为高细节模型"""
        if isinstance(mesh, dict):
            return mesh.get('is_high_detail', False)
        return getattr(mesh, 'is_high_detail', False)


class MaterialRecommender:
    """材料推荐引擎"""
    
    def __init__(self):
        self.analyzer = MeshAnalyzer()
        self.db = MATERIAL_DATABASE
    
    def recommend(self, mesh, target_size_mm: float) -> MaterialRecommendation:
        """
        根据3D模型特征推荐最适合的打印材料
        
        Args:
            mesh: 3D模型对象（支持dict或具有属性的对象）
            target_size_mm: 目标尺寸（mm），取最长边
            
        Returns:
            MaterialRecommendation: 包含推荐材料、备选材料和警告信息
        """
        warnings = []
        reasoning = []
        
        # 分析模型特征
        face_count = self.analyzer.get_face_count(mesh)
        max_overhang = self.analyzer.get_max_overhang_angle(mesh)
        wall_thickness = self.analyzer.get_wall_thickness(mesh)
        is_flexible = self.analyzer.is_flexible_model(mesh)
        is_high_detail = self.analyzer.is_high_detail(mesh)
        
        # 推荐逻辑
        recommended_name = self._determine_material(
            face_count, target_size_mm, max_overhang, 
            wall_thickness, is_flexible, is_high_detail,
            warnings, reasoning
        )
        
        recommended = self.db[recommended_name]
        alternatives = self._get_alternatives(recommended_name, face_count, target_size_mm)
        
        return MaterialRecommendation(
            recommended_material=recommended,
            alternative_materials=alternatives,
            warnings=warnings,
            reasoning=reasoning
        )
    
    def _determine_material(
        self, face_count: int, size: float, overhang: float,
        wall_thickness: float, is_flexible: bool, is_high_detail: bool,
        warnings: List[str], reasoning: List[str]
    ) -> str:
        """核心推荐逻辑"""
        
        # 规则1: 面数 > 100K 且壁厚 < 1mm → Resin
        if face_count > 100000 and wall_thickness < 1.0:
            reasoning.append(f"面数{face_count} > 100K 且壁厚{wall_thickness}mm < 1mm，需要高精度树脂打印")
            return "Resin"
        
        # 规则2: 柔性模型 → TPU
        if is_flexible:
            reasoning.append("模型标记为柔性件，推荐使用TPU材料")
            warnings.append("TPU打印速度较慢，需要调整打印参数")
            return "TPU"
        
        # 规则3: 高细节模型 → Resin
        if is_high_detail and face_count > 50000:
            reasoning.append("高细节模型，Resin可提供最佳表面质量")
            return "Resin"
        
        # 规则4: 有悬垂 > 60度 → PLA/PETG + 支撑提示
        if overhang > 60:
            reasoning.append(f"检测到悬垂角度{overhang}° > 60°，需要添加支撑结构")
            warnings.append("悬垂结构需要添加支撑，建议使用PLA或PETG以获得更好的支撑移除效果")
            # 对于大尺寸悬垂，选PETG；小尺寸选PLA
            if face_count > 50000:
                return "PETG"
            return "PLA"
        
        # 规则5: 尺寸 > 200mm → PETG（PLA翘曲风险）
        if size > 200:
            reasoning.append(f"模型尺寸{size}mm > 200mm，PETG比PLA更不易翘曲")
            warnings.append("大尺寸打印建议使用PETG，减少翘曲风险")
            return "PETG"
        
        # 规则6: 壁厚极薄 → Resin
        if wall_thickness < 0.5:
            reasoning.append(f"壁厚{wall_thickness}mm极薄，Resin可实现高精度打印")
            return "Resin"
        
        # 默认 → PLA
        reasoning.append("标准模型，使用最通用的PLA材料")
        return "PLA"
    
    def _get_alternatives(self, primary: str, face_count: int, size: float) -> List[Material]:
        """获取备选材料"""
        alternatives = []
        
        # 根据主推荐添加合理替代品
        if primary == "PLA":
            alternatives.append(self.db["PETG"])  # 更高强度
            alternatives.append(self.db["Resin"])  # 更高精度
        elif primary == "PETG":
            alternatives.append(self.db["ABS"])  # 更高耐热
            alternatives.append(self.db["PLA"])  # 更易打印
        elif primary == "ABS":
            alternatives.append(self.db["PETG"])  # 易打印
        elif primary == "TPU":
            alternatives.append(self.db["PETG"])  # 略硬
        elif primary == "Resin":
            if size > 100:
                alternatives.append(self.db["PLA"])  # 大尺寸用FDM
                alternatives.append(self.db["PETG"])
            else:
                alternatives.append(self.db["PLA"])
        
        return alternatives[:2]  # 最多返回2个替代品


def recommend(mesh, target_size_mm: float) -> MaterialRecommendation:
    """
    便捷函数：根据3D模型特征推荐最适合的打印材料
    
    Args:
        mesh: 3D模型对象（支持dict或具有属性的对象）
        target_size_mm: 目标尺寸（mm），取最长边
        
    Returns:
        MaterialRecommendation: 包含推荐材料、备选材料和警告信息
    """
    recommender = MaterialRecommender()
    return recommender.recommend(mesh, target_size_mm)
