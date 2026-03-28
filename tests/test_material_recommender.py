"""
测试材料推荐器
"""

import pytest
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from printforge.material_recommender import (
    MaterialRecommender,
    MaterialRecommendation,
    MATERIAL_DATABASE,
    recommend
)


class TestMaterialRecommender:
    """材料推荐器测试"""
    
    @pytest.fixture
    def recommender(self):
        return MaterialRecommender()
    
    def test_recommend_resin_for_high_poly_thin_wall(self, recommender):
        """测试：面数 > 100K 且壁厚 < 1mm → Resin"""
        mesh = {
            'face_count': 150000,
            'wall_thickness': 0.8,
            'max_overhang_angle': 30,
            'is_flexible': False,
            'is_high_detail': False
        }
        result = recommender.recommend(mesh, target_size_mm=50)
        
        assert result.recommended_material.name == "Resin"
        assert any("100K" in r for r in result.reasoning)
    
    def test_recommend_petg_for_large_size(self, recommender):
        """测试：尺寸 > 200mm → PETG（PLA翘曲风险）"""
        mesh = {
            'face_count': 50000,
            'wall_thickness': 2.0,
            'max_overhang_angle': 30,
            'is_flexible': False,
            'is_high_detail': False
        }
        result = recommender.recommend(mesh, target_size_mm=250)
        
        assert result.recommended_material.name == "PETG"
        assert any("200mm" in r for r in result.reasoning)
        assert any("翘曲" in w for w in result.warnings)
    
    def test_recommend_tpu_for_flexible_model(self, recommender):
        """测试：柔性模型 → TPU"""
        mesh = {
            'face_count': 30000,
            'wall_thickness': 3.0,
            'max_overhang_angle': 45,
            'is_flexible': True,
            'is_high_detail': False
        }
        result = recommender.recommend(mesh, target_size_mm=100)
        
        assert result.recommended_material.name == "TPU"
        assert any("柔性" in r for r in result.reasoning)
    
    def test_recommend_with_overhang_warning(self, recommender):
        """测试：有悬垂 > 60度 → PLA/PETG + 支撑提示"""
        mesh = {
            'face_count': 40000,
            'wall_thickness': 2.0,
            'max_overhang_angle': 70,
            'is_flexible': False,
            'is_high_detail': False
        }
        result = recommender.recommend(mesh, target_size_mm=100)
        
        # 应该推荐PLA（面数小于50K）
        assert result.recommended_material.name == "PLA"
        assert any("支撑" in w for w in result.warnings)
        assert any("60°" in r or "悬垂" in r for r in result.reasoning)
    
    def test_recommend_pla_as_default(self, recommender):
        """测试：默认 → PLA"""
        mesh = {
            'face_count': 30000,
            'wall_thickness': 2.0,
            'max_overhang_angle': 30,
            'is_flexible': False,
            'is_high_detail': False
        }
        result = recommender.recommend(mesh, target_size_mm=100)
        
        assert result.recommended_material.name == "PLA"
        assert any("通用" in r or "PLA" in r for r in result.reasoning)


class TestMaterialDatabase:
    """材料数据库测试"""
    
    def test_all_materials_exist(self):
        """测试所有必需材料都存在"""
        required = ["PLA", "PETG", "ABS", "TPU", "Resin"]
        for name in required:
            assert name in MATERIAL_DATABASE, f"材料 {name} 不存在"
    
    def test_material_attributes(self):
        """测试材料属性完整性"""
        for name, material in MATERIAL_DATABASE.items():
            assert hasattr(material, 'name')
            assert hasattr(material, 'temperature_nozzle')
            assert hasattr(material, 'temperature_bed')
            assert hasattr(material, 'bed_needed')
            assert hasattr(material, 'enclosure_needed')
            assert hasattr(material, 'description')
            assert hasattr(material, 'suitable_for')
            assert isinstance(material.suitable_for, list)


class TestConvenienceFunction:
    """便捷函数测试"""
    
    def test_recommend_function_works(self):
        """测试便捷函数返回正确类型"""
        mesh = {'face_count': 10000, 'wall_thickness': 2.0}
        result = recommend(mesh, target_size_mm=50)
        
        assert isinstance(result, MaterialRecommendation)
        assert result.recommended_material is not None
        assert isinstance(result.alternative_materials, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.reasoning, list)
