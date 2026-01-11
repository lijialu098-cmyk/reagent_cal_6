import streamlit as st
import re
import pandas as pd
from PIL import Image
import numpy as np
import os
import sys

# 显示当前Python解释器路径
st.write(f"Python解释器路径: {sys.executable}")

# 图像处理和OCR
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    # 检查Tesseract是否安装
    import subprocess
    try:
        subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
        TESSERACT_AVAILABLE = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        TESSERACT_AVAILABLE = False
except ImportError:
    TESSERACT_AVAILABLE = False

# Cropper
try:
    from streamlit_cropper import st_cropper
    CROPPER_AVAILABLE = True
except ImportError:
    CROPPER_AVAILABLE = False


class StreamlitCalculator:
    def __init__(self):
        if "stock_solutions" not in st.session_state:
            self.init_stock_solutions()
        if "molecular_weights" not in st.session_state:
            self.init_mw()
        if "extracted_text" not in st.session_state:
            st.session_state.extracted_text = ""
        if "uploaded_image" not in st.session_state:
            st.session_state.uploaded_image = None
        if "processed_image" not in st.session_state:
            st.session_state.processed_image = None
        if "calculation_results" not in st.session_state:
            st.session_state.calculation_results = None
        if "total_volume_ml" not in st.session_state:
            st.session_state.total_volume_ml = 1000.0
        if "show_processed_image" not in st.session_state:
            st.session_state.show_processed_image = False

    # ------------------------
    # 初始化
    # ------------------------
    def init_stock_solutions(self):
        st.session_state.stock_solutions = {
            "Tris": {"concentration": 2.0, "unit": "M", "density": 1.0},
            "NaCl": {"concentration": 5.0, "unit": "M", "density": 1.0},
            "甘油": {"concentration": 100.0, "unit": "%", "density": 1.26},
            "Trehalose": {"concentration": 40.0, "unit": "%", "density": 1.0},
            "DTT": {"concentration": 1.0, "unit": "M", "density": 1.0},
            "NaAc": {"concentration": 1.0, "unit": "M", "density": 1.0},
            "PBS": {"concentration": 10.0, "unit": "X", "density": 1.0},
            "Brij-35": {"concentration": 5.0, "unit": "%", "density": 1.0},
            "IMI": {"concentration": 2.0, "unit": "M", "density": 1.0},
            "HEPES": {"concentration": 1.0, "unit": "M", "density": 1.0},
            "MES": {"concentration": 2.0, "unit": "M", "density": 1.0},
            "EDTA": {"concentration": 200.0, "unit": "mM", "density": 1.0},
            "NH4PO4": {"concentration": 3.0, "unit": "M", "density": 1.0},
            "CHAPS": {"concentration": 10.0, "unit": "%", "density": 1.0},
        }

    def init_mw(self):
        st.session_state.molecular_weights = {
            "Tris": 121.14,
            "CHAPS": 614.88,
        }

    # ======================================================
    # OCR 处理 - 使用新版本的完整流程
    # ======================================================
    def preprocess_image_for_ocr(self, image):
        """
        使用新版本的图像预处理
        灰度图 -> 自适应阈值二值化 -> 形态学操作
        """
        if not CV2_AVAILABLE:
            st.error("OpenCV (cv2) 未安装，无法进行图像处理")
            return None
        
        try:
            # 转换为OpenCV格式 (BGR)
            if isinstance(image, Image.Image):
                img = np.array(image.convert("RGB"))
                # PIL是RGB，OpenCV需要BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif isinstance(image, np.ndarray):
                img = image.copy()
            else:
                return None
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 自适应阈值二值化
            gray = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31, 5
            )
            
            # 形态学操作（闭运算）增强字符
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            return gray
            
        except Exception as e:
            st.error(f"图像预处理失败: {str(e)}")
            return None
    
    def extract_text_from_image(self, image):
        """
        使用新版本的OCR提取文本
        修改：强制显示预处理后的图像，并使用该图像进行OCR
        """
        # 添加assert语句检查cv2是否可用
        assert CV2_AVAILABLE, "cv2 未正确安装，OCR 预处理无法工作"
        
        if not TESSERACT_AVAILABLE:
            st.warning("Tesseract OCR 未安装或不可用，无法进行 OCR")
            return ""
        
        if not CV2_AVAILABLE:
            st.warning("OpenCV (cv2) 未安装，无法进行图像预处理")
            return ""
        
        try:
            # 图像预处理
            gray = self.preprocess_image_for_ocr(image)
            if gray is None:
                return ""
            
            # 保存处理后的图像到session state
            st.session_state.processed_image = gray
            
            # 强制显示预处理后的图像（修改点1）
            st.image(gray, caption="输入图像", clamp=True)
            
            # Tesseract配置 - 使用psm 7和新的配置（修改点）
            config = r'''--oem 3
--psm 7
-c preserve_interword_spaces=1
'''
            
            # 文字识别 - 必须使用gray而不是原始image（修改点2）
            text = pytesseract.image_to_string(gray, config=config)
            
            # 显示原始OCR结果（新增）
            st.write("RAW OCR:", repr(text))
            
            # 后处理：清理和格式化文本
            text = self.clean_ocr_text(text)
            
            return text.strip()
            
        except Exception as e:
            st.error(f"OCR处理失败: {str(e)}")
            return ""
    
    def clean_ocr_text(self, text):
        """
        清理OCR识别结果，提取配方信息
        """
        if not text:
            return ""

        # 统一符号
        text = text.replace('\n', ' ')
        text = re.sub(r'[，；、]', ',', text)

        # 先按逗号拆分组分
        parts = [p.strip() for p in text.split(',') if p.strip()]

        cleaned = []

        for part in parts:
            # 数字 + 单位 + 名称
            m = re.search(
                r'(\d+\.?\d*)\s*(mM|M|%|μM|uM)?\s*([A-Za-z\u4e00-\u9fa5][A-Za-z\u4e00-\u9fa5\s\-]*)',
                part,
                re.IGNORECASE
            )
            if m:
                value = m.group(1)
                unit = m.group(2) or "mM"
                name = m.group(3).strip()
                unit = self.normalize_unit(unit)
                cleaned.append(f"{value} {unit} {name}")
            else:
                cleaned.append(part)

        return "\n".join(cleaned)
    
    def normalize_unit(self, unit):
        """规范化单位"""
        unit = unit.strip().upper()
        unit_map = {
            'MM': 'mM',
            'ΜM': 'μM',
            'UM': 'μM',
            'U M': 'μM',
            'M M': 'mM',
            '%': '%',
            'X': 'X',
            'M': 'M'
        }
        return unit_map.get(unit, unit)

    # ------------------------
    # 解析和计算核心
    # ------------------------
    def parse_formula_string(self, formula_input):
        """解析配方字符串，支持多种格式"""
        formula_input = formula_input.strip()
        formula_input = re.sub(r'[，；、]', ',', formula_input)
        
        # 正则表达式模式：匹配 数字+可选小数+可选空格+单位+可选空格+名称
        pattern = r'([\d\.]+)\s*([mMuM%Xxμµ]*)\s*([a-zA-Z\u4e00-\u9fa5\-]+(?:\s*[a-zA-Z\u4e00-\u9fa5\-]+)*)'
        
        matches = re.findall(pattern, formula_input)
        components = {}
        
        for match in matches:
            if not match:
                continue
                
            value_str = match[0].strip()
            unit = match[1].strip()
            name = match[2].strip()
            
            # 处理没有明确单位的情况
            if not unit:
                name_match = re.match(r'^([mMuMμµ])([a-zA-Z\u4e00-\u9fa5].*)', name)
                if name_match:
                    unit = name_match.group(1)
                    name = name_match.group(2)
                else:
                    unit = "mM"
            
            try:
                value = float(value_str)
            except ValueError:
                continue
            
            # 单位规范化
            unit = unit.upper()
            if unit in ["M", "Μ"]:
                unit = "M"
            elif unit in ["MM", "mM", "ΜM", "µM"]:
                unit = "mM"
            elif unit == "%":
                unit = "%"
            elif unit == "X":
                unit = "X"
            elif unit in ["UM", "μM", "ΜM", "UM", "µM"]:
                unit = "μM"
            
            # 名称规范化
            name = self.normalize_name(name)
            
            if name:
                components[name] = {
                    'target_concentration': value,
                    'target_unit': unit
                }
        
        return components

    def normalize_name(self, name):
        """规范化组分名称"""
        name_mapping = {
            "Tris": "Tris", "TRIS": "Tris", "tris": "Tris", "Tirs": "Tris", "Tns": "Tris",
            "NaCl": "NaCl", "NACL": "NaCl", "nacl": "NaCl", "NaCI": "NaCl", "氯化钠": "NaCl",
            "甘油": "甘油", "Glycerol": "甘油", "glycerol": "甘油", "olycerol": "甘油",
            "Trehalose": "Trehalose", "trehalose": "Trehalose", "海藻糖": "Trehalose",
            "DTT": "DTT", "dtt": "DTT",
            "PBS": "PBS", "pbs": "PBS",
            "HEPES": "HEPES", "hepes": "HEPES",
            "MES": "MES", "mes": "MES",
            "EDTA": "EDTA", "edta": "EDTA",
            "IMI": "IMI", "imi": "IMI", "咪唑": "IMI",
            "Brij-35": "Brij-35", "brij-35": "Brij-35",
            "NaAc": "NaAc", "NaOAc": "NaAc", "naac": "NaAc", "乙酸钠": "NaAc",
            "NH4PO4": "NH4PO4", "nh4po4": "NH4PO4", "磷酸铵": "NH4PO4",
            "CHAPS": "CHAPS", "chaps": "CHAPS", "卡普斯": "CHAPS",
        }
        
        for key in name_mapping:
            if name.lower() == key.lower():
                return name_mapping[key]
        
        # 模糊匹配
        for key in name_mapping:
            if self.is_similar_name(name, key, threshold=0.7):
                return name_mapping[key]
        
        return name

    def is_similar_name(self, name1, name2, threshold=0.7):
        """检查两个名称是否相似"""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        if name1_lower == name2_lower:
            return True
        
        # 简单相似度计算
        if len(name1_lower) < 3 or len(name2_lower) < 3:
            return name1_lower in name2_lower or name2_lower in name1_lower
        
        common_chars = set(name1_lower) & set(name2_lower)
        similarity = len(common_chars) / max(len(set(name1_lower)), len(set(name2_lower)))
        return similarity > threshold

    def parse_volume(self, volume_str):
        """解析体积字符串"""
        try:
            volume_str = volume_str.strip().lower()
            match = re.match(r'([\d\.]+)\s*([a-zA-Zμµ]*)?', volume_str)
            if not match:
                return None
                
            number = float(match.group(1))
            unit = match.group(2) if match.group(2) else ""
            
            if unit in ['l', '升', 'liter', 'litre']:
                return number * 1000
            elif unit in ['ml', '毫升', 'milliliter']:
                return number
            elif unit in ['ul', 'μl', 'µl', '微升', 'microliter']:
                return number / 1000
            else:
                # 默认单位为毫升
                return number
        except Exception:
            return None

    def calculate_component_volume(self, stock_concentration, stock_unit, 
                                  target_concentration, target_unit, 
                                  total_volume_ml, component_name):
        """计算单个组分的体积"""
        # 特殊处理CHAPS：10%母液到mM的转换
        if component_name == "CHAPS" and stock_unit == "%" and target_unit == "mM":
            # CHAPS分子量
            mw = st.session_state.molecular_weights.get("CHAPS", 614.88)
            
            # 10% CHAPS的摩尔浓度计算
            # 10% = 10g/100mL = 100g/L
            # 摩尔浓度 = 100g/L ÷ 614.88g/mol = 0.1626 M = 162.6 mM
            stock_molarity = (stock_concentration * 10) / mw  # 转换为M
            stock_mm = stock_molarity * 1000  # 转换为mM
            
            # 计算体积：V1 = (C2 * V2) / C1
            volume_ml = (target_concentration * total_volume_ml) / stock_mm
            
            # 计算所需CHAPS质量（用于验证）
            required_mol = (target_concentration / 1000) * (total_volume_ml / 1000)
            required_mass_g = required_mol * mw
            
            # CHAPS母液需要称量的质量 = 所需CHAPS质量 ÷ 10%
            stock_solution_mass = required_mass_g / (stock_concentration / 100)
            
            return volume_ml, stock_solution_mass
        
        # 特殊处理CHAPS：10%母液到μM的转换
        if component_name == "CHAPS" and stock_unit == "%" and target_unit == "μM":
            # CHAPS分子量
            mw = st.session_state.molecular_weights.get("CHAPS", 614.88)
            
            # 10% CHAPS的摩尔浓度计算
            # 10% = 10g/100mL = 100g/L
            # 摩尔浓度 = 100g/L ÷ 614.88g/mol = 0.1626 M = 162.6 mM = 162600 μM
            stock_molarity = (stock_concentration * 10) / mw  # 转换为M
            stock_um = stock_molarity * 1000000  # 转换为μM
            
            # 计算体积：V1 = (C2 * V2) / C1
            volume_ml = (target_concentration * total_volume_ml) / stock_um
            
            # 计算所需CHAPS质量（用于验证）
            required_mol = (target_concentration / 1000000) * (total_volume_ml / 1000)
            required_mass_g = required_mol * mw
            
            # CHAPS母液需要称量的质量 = 所需CHAPS质量 ÷ 10%
            stock_solution_mass = required_mass_g / (stock_concentration / 100)
            
            return volume_ml, stock_solution_mass
        
        # 将目标浓度转换为与母液相同的单位
        target_value_in_stock_unit = self.convert_to_stock_unit(
            target_concentration, target_unit, stock_unit, component_name
        )
        
        if target_value_in_stock_unit is None:
            return None, None
        
        # 计算体积：C1V1 = C2V2
        if stock_concentration > 0:
            volume = (target_value_in_stock_unit * total_volume_ml) / stock_concentration
            
            # 计算母液需要称量的质量 = 体积 × 密度
            density = st.session_state.stock_solutions.get(component_name, {}).get("density", 1.0)
            mass = volume * density
            
            return volume, mass
        return 0, 0

    def convert_to_stock_unit(self, target_value, target_unit, stock_unit, component_name):
        """将目标浓度转换为母液单位"""
        if target_unit == stock_unit:
            return target_value
        
        # 处理百分比和倍浓度
        if target_unit == '%' and stock_unit == '%':
            return target_value
        if target_unit == 'X' and stock_unit == 'X':
            return target_value
        
        # 摩尔浓度转换
        conversions = {
            ('M', 'mM'): 1000,
            ('M', 'μM'): 1000000,
            ('mM', 'M'): 0.001,
            ('mM', 'μM'): 1000,
            ('μM', 'M'): 0.000001,
            ('μM', 'mM'): 0.001,
            ('μM', 'μM'): 1,  # 同单位
        }
        
        key = (target_unit, stock_unit)
        if key in conversions:
            return target_value * conversions[key]
        
        # 对于百分比和摩尔浓度之间的转换，需要分子量
        if (target_unit == '%' and stock_unit in ['M', 'mM', 'μM']) or \
           (stock_unit == '%' and target_unit in ['M', 'mM', 'μM']):
            mw = st.session_state.molecular_weights.get(component_name)
            if mw:
                if target_unit == '%':
                    # % 转换为 M: 10% = 100g/L = 100/mw M
                    target_M = (target_value * 10) / mw
                    return self.convert_to_stock_unit(target_M, 'M', stock_unit, component_name)
                else:
                    # M 转换为 %
                    target_M = self.convert_to_stock_unit(target_value, target_unit, 'M', component_name)
                    target_percent = (target_M * mw) / 10
                    return target_percent
        
        return target_value

    # ------------------------
    # 主计算函数
    # ------------------------
    def calculate_volumes(self, components, total_volume_ml):
        """计算各组分体积"""
        results = {
            'components': {},
            'total_stock_volume': 0,
            'water_volume': 0,
        }
        
        for name, info in components.items():
            # 检查是否有母液
            if name in st.session_state.stock_solutions:
                stock = st.session_state.stock_solutions[name]
                target_value = info['target_concentration']
                target_unit = info['target_unit']
                
                # 计算体积和质量
                volume, mass = self.calculate_component_volume(
                    stock_concentration=stock['concentration'],
                    stock_unit=stock['unit'],
                    target_concentration=target_value,
                    target_unit=target_unit,
                    total_volume_ml=total_volume_ml,
                    component_name=name
                )
                
                if volume is None:
                    continue
                
                results['components'][name] = {
                    'stock_concentration': f"{stock['concentration']}{stock['unit']}",
                    'stock_volume_ml': volume,
                    'mass_g': mass if mass is not None else 0,
                    'target_concentration': target_value,
                    'target_unit': target_unit,
                    'needs_weighing': (mass is not None and volume == 0)
                }
                
                results['total_stock_volume'] += volume
            # 检查是否有分子量（直接称量）
            elif name in st.session_state.molecular_weights:
                mw = st.session_state.molecular_weights[name]
                target_value = info['target_concentration']
                target_unit = info['target_unit']
                
                # 将目标浓度转换为M
                if target_unit == 'mM':
                    target_M = target_value / 1000
                elif target_unit == 'μM':
                    target_M = target_value / 1000000
                elif target_unit == 'M':
                    target_M = target_value
                else:
                    st.warning(f"无法处理 {name} 的单位: {target_unit}")
                    continue
                
                # 计算物质的量和质量
                mol = target_M * (total_volume_ml / 1000)
                mass = mol * mw
                
                results['components'][name] = {
                    'stock_concentration': "N/A",
                    'stock_volume_ml': 0,
                    'mass_g': mass,
                    'target_concentration': target_value,
                    'target_unit': target_unit,
                    'needs_weighing': True
                }
            else:
                st.warning(f"未找到 {name} 的母液或分子量信息")
        
        results['water_volume'] = total_volume_ml - results['total_stock_volume']
        if results['water_volume'] < 0:
            st.error("错误：母液总体积超过了目标体积")
            return None
        
        return results

    # ------------------------
    # 显示结果
    # ------------------------
    def show_results(self, results, total_ml):
        """显示计算结果"""
        st.header("📊 计算结果")
        
        # 创建结果表格
        df_data = []
        for name, comp in results['components'].items():
            if comp['stock_volume_ml'] > 0:
                df_data.append({
                    "组分": name,
                    "目标浓度": f"{comp['target_concentration']} {comp['target_unit']}",
                    "母液": comp['stock_concentration'],
                    "体积(mL)": f"{comp['stock_volume_ml']:.2f}",
                    "质量(g)": f"{comp['mass_g']:.2f}"
                })
            elif comp['needs_weighing']:
                df_data.append({
                    "组分": name,
                    "目标浓度": f"{comp['target_concentration']} {comp['target_unit']}",
                    "母液": "直接称量",
                    "体积(mL)": "-",
                    "质量(g)": f"{comp['mass_g']:.2f}"
                })
        
        # 添加水
        if results['water_volume'] > 0:
            df_data.append({
                "组分": "水",
                "目标浓度": "-",
                "母液": "-",
                "体积(mL)": f"{results['water_volume']:.2f}",
                "质量(g)": f"{results['water_volume']:.2f}"
            })
        
        df = pd.DataFrame(df_data)
        
        # 显示表格
        st.dataframe(df, use_container_width=True, hide_index=True)
      
    # ------------------------
    # UI界面
    # ------------------------
    def calculate(self):
        st.title("🧪 试剂配方计算器")
        st.caption("拍照识别中文时会报错，但可在输入框手动修改")
        
        # 显示OCR状态
        if not TESSERACT_AVAILABLE or not CV2_AVAILABLE:
            st.warning("⚠️ OCR功能可能不可用")
            if not TESSERACT_AVAILABLE:
                st.error("Tesseract OCR未安装")
            if not CV2_AVAILABLE:
                st.error("OpenCV (cv2) 未安装")
        
        # 创建标签页
        tab1, tab2 = st.tabs(["📝 配方计算", "⚙️ 母液管理"])
        
        with tab1:
            # 输入区域
            with st.container():
                # 重新开始按钮
                if st.button("🔄 重新开始", type="secondary", use_container_width=False):
                    st.session_state.calculation_results = None
                    st.session_state.extracted_text = ""
                    st.session_state.uploaded_image = None
                    st.session_state.processed_image = None
                    st.rerun()
                
                method = st.radio("输入方式", ["手动输入", "拍照识别"], horizontal=True)
                
                formula_input = ""
                volume_input = "1 L"
                
                # 手动输入
                if method == "手动输入":
                    formula_input = st.text_area(
                        "配方输入",
                        "20 mM Tris\n150 mM NaCl\n20% 甘油\n11% Trehalose\n1 mM DTT\n33 μM CHAPS",
                        height=150,
                        help="示例格式: 20 mM Tris, 150 mM NaCl, 20% 甘油, 11% Trehalose, 1 mM DTT, 33 μM CHAPS"
                    )
                
                # OCR识别
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        uploaded = st.file_uploader("上传实验笔记图片", type=["png", "jpg", "jpeg"], 
                                                  key="ocr_uploader")
                    
                    with col2:
                        if uploaded:
                            st.session_state.uploaded_image = Image.open(uploaded)
                            st.image(st.session_state.uploaded_image, caption="上传的图片", use_container_width=True)
                    
                    if st.session_state.get("uploaded_image"):
                        if CROPPER_AVAILABLE:
                            st.write("📐 裁剪识别区域 (可选)")
                            cropped_img = st_cropper(st.session_state.uploaded_image, 
                                                   realtime_update=True, 
                                                   box_color='#00FF00',
                                                   aspect_ratio=None)
                            
                            if cropped_img is not None:
                                # st_cropper 返回的是PIL Image
                                if isinstance(cropped_img, np.ndarray):
                                    cropped_img = Image.fromarray(cropped_img.astype('uint8'), 'RGB')
                                st.image(cropped_img, caption="裁剪后的图片", use_container_width=True)
                                img_to_process = cropped_img
                            else:
                                img_to_process = st.session_state.uploaded_image
                        else:
                            img_to_process = st.session_state.uploaded_image
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔍 开始识别图片", type="primary", use_container_width=True):
                                with st.spinner("正在识别文字..."):
                                    # 注意：现在extract_text_from_image内部会强制显示预处理后的图像
                                    extracted_text = self.extract_text_from_image(img_to_process)
                                    if extracted_text and extracted_text.strip():
                                        st.session_state.extracted_text = extracted_text
                                        st.success("识别完成！")
                                    else:
                                        st.warning("未能识别出有效文字，请手动修改或重新上传")
                                        st.session_state.extracted_text = ""
                        
                        with col2:
                            # 显示处理后的图像选项（作为额外查看，主要图像已在OCR过程中显示）
                            st.session_state.show_processed_image = st.checkbox("再次查看处理后的图像", 
                                                                               value=st.session_state.show_processed_image)
                        
                        # 显示处理后的图像（仅当复选框选中时）
                        if st.session_state.show_processed_image and st.session_state.processed_image is not None:
                            st.image(st.session_state.processed_image, caption="预处理后的图像（再次查看）", 
                                    use_container_width=True, clamp=True)
                        
                        # 显示识别结果
                        if st.session_state.extracted_text:
                            formula_input = st.text_area("识别 / 输入的配方文本", 
                                                        st.session_state.extracted_text, 
                                                        height=150,
                                                        key="ocr_result")
                        else:
                            formula_input = st.session_state.get("extracted_text", "")
                
                # 体积输入和计算按钮
                col1, col2 = st.columns([2, 1])
                with col1:
                    volume_input = st.text_input("目标体积", "1 L", 
                                               help="支持单位: L(升), mL(毫升), μL(微升)")
                
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    calculate_clicked = st.button("🚀 开始计算", type="primary", use_container_width=True)
                
                # 处理计算逻辑
                if calculate_clicked:
                    if not formula_input:
                        st.error("请输入配方")
                    else:
                        try:
                            # 解析体积
                            total_ml = self.parse_volume(volume_input)
                            if not total_ml:
                                st.error("请输入有效的体积")
                            else:
                                # 保存体积到session state
                                st.session_state.total_volume_ml = total_ml
                                
                                # 解析配方
                                components = self.parse_formula_string(formula_input)
                                if not components:
                                    st.error("无法解析配方")
                                else:
                                    # 显示解析结果
                                    with st.expander("📝 解析到的组分", expanded=False):
                                        col1, col2 = st.columns(2)
                                        component_list = list(components.items())
                                        half = len(component_list) // 2 + len(component_list) % 2
                                        
                                        with col1:
                                            for name, info in component_list[:half]:
                                                st.write(f"• {name}: {info['target_concentration']} {info['target_unit']}")
                                        with col2:
                                            for name, info in component_list[half:]:
                                                st.write(f"• {name}: {info['target_concentration']} {info['target_unit']}")
                                    
                                    # 计算
                                    results = self.calculate_volumes(components, total_ml)
                                    if results:
                                        st.session_state.calculation_results = results
                                        st.rerun()
                                    
                        except Exception as e:
                            st.error(f"计算失败: {str(e)}")
                
                # 显示计算结果（如果有）
                if st.session_state.calculation_results:
                    st.markdown("---")
                    self.show_results(st.session_state.calculation_results, st.session_state.total_volume_ml)
        
        with tab2:
            self.manage_stocks()

    # ------------------------
    # 母液管理
    # ------------------------
    def manage_stocks(self):
        """管理母液库"""
        st.header("⚙️ 母液管理")
        
        # 显示当前母液
        if st.session_state.stock_solutions:
            stock_data = []
            for name, sol in st.session_state.stock_solutions.items():
                stock_data.append({
                    "名称": name,
                    "浓度": f"{sol['concentration']} {sol['unit']}",
                    "密度": f"{sol['density']} g/mL"
                })
            
            df_stocks = pd.DataFrame(stock_data)
            st.dataframe(df_stocks, use_container_width=True, hide_index=True)
        else:
            st.info("暂无母液数据")
        
        # 添加新母液
        with st.expander("➕ 添加新母液", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("名称", key="new_stock_name")
                new_unit = st.selectbox("单位", ["M", "mM", "μM", "%", "X", "mg/mL", "g/L"], 
                                      key="new_stock_unit")
            with col2:
                new_conc = st.number_input("浓度", value=1.0, min_value=0.0, step=0.1, 
                                         format="%.3f", key="new_stock_conc")
                new_density = st.number_input("密度 (g/mL)", value=1.0, min_value=0.1, 
                                            max_value=10.0, step=0.1, key="new_stock_density")
            
            col3, col4 = st.columns(2)
            with col3:
                new_mw = st.number_input("分子量 (可选)", value=0.0, min_value=0.0, 
                                       step=0.01, format="%.4f", key="new_stock_mw")
            with col4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✅ 添加母液", type="primary", use_container_width=True):
                    if new_name and new_name.strip():
                        st.session_state.stock_solutions[new_name] = {
                            "concentration": new_conc,
                            "unit": new_unit,
                            "density": new_density
                        }
                        if new_mw > 0:
                            st.session_state.molecular_weights[new_name] = new_mw
                        st.success(f"已添加 {new_name}")
                        st.rerun()
                    else:
                        st.error("请输入母液名称")
        
        # 删除母液
        with st.expander("🗑️ 删除母液", expanded=True):
            if st.session_state.stock_solutions:
                delete_name = st.selectbox("选择要删除的母液", 
                                         list(st.session_state.stock_solutions.keys()),
                                         key="delete_select")
                
                if st.button("❌ 删除选中母液", type="secondary", use_container_width=True):
                    if delete_name in st.session_state.stock_solutions:
                        del st.session_state.stock_solutions[delete_name]
                        if delete_name in st.session_state.molecular_weights:
                            del st.session_state.molecular_weights[delete_name]
                        st.success(f"已删除 {delete_name}")
                        st.rerun()
            else:
                st.info("没有可删除的母液")


# ------------------------
# 主入口
# ------------------------
if __name__ == "__main__":
    st.set_page_config(
        page_title="试剂配方计算器",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo',
            'Report a bug': "https://github.com/your-repo/issues",
            'About': "## 🧪 试剂配方计算器\n\n使用Tesseract OCR支持手写配方识别和μM单位计算的配方计算工具"
        }
    )
    
    # 添加侧边栏
    with st.sidebar:
        st.title("状态检验")
        
        # 显示Python解释器路径
        st.write(f"Python路径: {sys.executable}")
        st.write(f"建议在命令行运行: `bash which python` 查看当前Python解释器")
        
        # 显示OCR状态
        if TESSERACT_AVAILABLE:
            st.success("✅ Tesseract OCR可用")
        else:
            st.error("❌ Tesseract OCR不可用")
        
        if CV2_AVAILABLE:
            st.success("✅ OpenCV可用")
        else:
            st.error("❌ OpenCV不可用")
        
        st.markdown("---")
        st.header("📖 使用说明")
        with st.expander("查看详细说明", expanded=True):
            st.markdown("""
            ### 注意：
            **手动输入格式**：
                    单行，多行
                    空格，逗号，
            
            **母液管理**：修改母液成分，仅对本次有效，再次打开消失。
                         后续会更新版本
         
            ### 🔢 单位支持：
            - **浓度**：M, mM, μM, %, X
            - **体积**：L, mL, μL
            
            ### ⚗️ 计算原理：
            1. C1V1 = C2V2（母液稀释）
            2. 考虑试剂密度计算质量
            3. CHAPS通过分子量614.88计算
            """)
        
        st.markdown("---")
        st.header("快速操作")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 重置所有", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        st.markdown("---")
        st.caption("版本 2.0 • 使用Tesseract OCR")
    
    # 主计算界面
    app = StreamlitCalculator()
    app.calculate()