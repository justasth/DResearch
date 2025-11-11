import folium
import geopandas as gpd
import pandas as pd
import numpy as np
import os
from branca.colormap import linear
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import Point
import webbrowser
from folium import plugins

# 定义象限分类函数
def classify_quadrant(dpl, dpe, dle):
    """
    根据Dpl、Dpe、Dle的值分类象限
    返回象限名称
    注意：处理边界情况（等于0的情况和浮点数精度问题）
    """
    # 处理NaN值
    if pd.isna(dpl) or pd.isna(dpe) or pd.isna(dle):
        return "未分类"
    
    # 使用小的阈值处理浮点数精度问题（避免因为精度问题导致等于0的判断失败）
    epsilon = 1e-10
    
    # 重新计算Dle以确保一致性（防止数据不一致）
    calculated_dle = dpl - dpe
    # 如果计算的Dle与提供的Dle差异很大，使用计算的Dle
    if abs(calculated_dle - dle) > 0.01:
        dle = calculated_dle
    
    # 使用阈值判断正负（处理浮点数精度问题）
    dpl_pos = dpl > epsilon
    dpl_neg = dpl < -epsilon
    dpe_pos = dpe > epsilon
    dpe_neg = dpe < -epsilon
    dle_pos = dle > epsilon
    dle_neg = dle < -epsilon
    dle_zero = abs(dle) <= epsilon
    
    # 象限1-1: Dpl>0, Dpe>0, Dle<0
    if dpl_pos and dpe_pos and dle_neg:
        return "象限1-1"
    # 象限1-2: Dpl>0, Dpe>0, Dle>0
    elif dpl_pos and dpe_pos and dle_pos:
        return "象限1-2"
    # 象限2: Dpl<0, Dpe>0, Dle>0 或 Dpl<0, Dpe>0, Dle<0
    elif dpl_neg and dpe_pos:
        return "象限2"
    # 象限3-1: Dpl<0, Dpe<0, Dle>0
    elif dpl_neg and dpe_neg and dle_pos:
        return "象限3-1"
    # 象限3-2: Dpl<0, Dpe<0, Dle<0
    elif dpl_neg and dpe_neg and dle_neg:
        return "象限3-2"
    # 象限4: Dpl>0, Dpe<0, Dle<0 或 Dpl>0, Dpe<0, Dle>0
    elif dpl_pos and dpe_neg:
        return "象限4"
    else:
        # 根据Dpl和Dpe的主要符号进行默认分类
        if dpl_pos and dpe_pos:
            return "象限1-2"  # 默认
        elif dpl_neg and dpe_neg:
            return "象限3-1"  # 默认
        elif dpl_neg and dpe_pos:
            return "象限2"
        elif dpl_pos and dpe_neg:
            return "象限4"
        else:
            return "象限1-2"  # 最终默认

# 定义Dpl的自定义区间和颜色
# 值越小，颜色越淡（接近白色）；值越大，颜色越深
DPL_INTERVALS = [-4, -2, 0, 2, 4, 6]  # 6个边界点定义5个区间
# 从最浅到最深：最小区间接近白色，最大区间最深
DPL_COLORS = ['#FFF5E6', '#FFE5CC', '#FFCC99', '#FF9900', '#CC6600']  # 对应5个区间，从浅到深

# 定义Dpe的自定义区间和颜色
# 值越小，颜色越淡（接近白色）；值越大，颜色越深
DPE_INTERVALS = [-15, -4, -2, 0, 2, 4]  # 6个边界点定义5个区间
# 从最浅到最深：最小区间接近白色，最大区间最深
DPE_COLORS = ['#F5F5FF', '#E5E5FF', '#CCCBFF', '#9999FF', '#6666FF']  # 对应5个区间，从浅到深

# 配置参数
SHP_PATH = "data/长三角市级边界.shp"
EXCEL_PATH  = "analysisdata.xlsx"
OUTPUT_HTML = "1990长三角人地产结构偏离度地图.html"
OUTPUT_PNG = "1990长三角人地产结构偏离度地图.png"
DATA_COLUMN = "Dpl1990"  # 要可视化的数据列
DATA_COLUMN2 = "Dpe1990"  # 要可视化的数据列
USE_BIVARIATE = True  # 是否使用双变量映射

# 统一的坐标轴范围（确保三个地图对齐）
UNIFIED_X_MIN = 114.878463
UNIFIED_Y_MIN = 27.143423
UNIFIED_X_MAX = 122.834203
UNIFIED_Y_MAX = 35.127197

# 1. 加载行政边界数据
print("\n[步骤1/5] 加载行政边界数据...")
gdf = gpd.read_file(SHP_PATH, engine='fiona', encoding='utf-8')

# 检查坐标参考系统(CRS)，如果不是WGS84(EPSG:4326)则转换
if gdf.crs is None or (hasattr(gdf.crs, 'to_epsg') and gdf.crs.to_epsg() != 4326):
    print("   - 转换坐标系到WGS84 (EPSG:4326)...")
    gdf = gdf.to_crs(epsg=4326)

print(f"   - 边界数据加载完成，包含 {len(gdf)} 个要素")
print(f"   - 字段列表: {gdf.columns.tolist()}")
# 打印示例城市名称
if '地名' in gdf.columns:
    print(f"   - 城市名称示例: {gdf['地名'].head(3).tolist()}")

# 2. 加载Excel数据
print("\n[步骤2/5] 加载Excel数据...")
df = pd.read_excel(EXCEL_PATH)
print(f"   - Excel数据加载完成，包含 {len(df)} 行")
print(f"   - 列名: {df.columns.tolist()}")

# 3. 合并地理数据与Excel数据
print("\n[步骤3/5] 合并地理数据和Excel数据...")
print(f"   - 使用SHP的'地名'字段和Excel的'全称'字段进行匹配")

merged_gdf = gdf.merge(df, left_on='地名', right_on='全称', how='left')

# 检查合并结果
print(f"   - 合并后记录数: {len(merged_gdf)}")
matched_count = merged_gdf[DATA_COLUMN].notna().sum()
print(f"   - 成功匹配: {matched_count}/{len(merged_gdf)}")

# 检查缺失数据
missing_data = merged_gdf[merged_gdf[DATA_COLUMN].isna()]
if not missing_data.empty:
    print(f"   - 警告: {len(missing_data)} 个城市没有匹配到数据:")
    for idx, row in missing_data.head(10).iterrows():
        print(f"      - {row.get('地名', 'N/A')} (SHP名称) / {row.get('全称', 'N/A')} (Excel名称)")

# 4. 创建地图
print("\n[步骤4/5] 创建交互式地图...")

# 计算地图中心点
x_min, y_min, x_max, y_max = merged_gdf.total_bounds
center_y = (y_min + y_max) / 2
center_x = (x_min + x_max) / 2

# 初始化地图
m = folium.Map(
    location=[center_y, center_x],
    zoom_start=7,
    tiles='https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
    attr='高德地图',
    control_scale=True
)

# 添加底图选项
folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(m)
try:
    folium.TileLayer('Stamen Terrain', name='地形图', attr='Stamen Terrain').add_to(m)
except:
    pass
folium.TileLayer('cartodbpositron', name='浅色底图').add_to(m)

# 颜色混合函数：将两种颜色混合
def hex_to_rgb(hex_color):
    """将十六进制颜色转换为RGB元组"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """将RGB元组转换为十六进制颜色"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def blend_colors_multiply(color1, color2):
    """使用Multiply混合模式：光与光的相乘混合
    特点：两个浅色混合变得更浅（接近白色），两个深色混合变得更深
    适合：值越小颜色越淡的场景，叠加后小值组合会显示为更浅的颜色
    """
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    # Multiply: result = (color1 * color2) / 255
    # 白色(255,255,255) * 任何颜色 = 那个颜色本身
    # 两个浅色相乘会得到更浅的结果
    blended = tuple(
        int((rgb1[i] * rgb2[i]) / 255)
        for i in range(3)
    )
    return rgb_to_hex(blended)

def blend_colors_screen(color1, color2):
    """使用Screen混合模式：光与光的相加混合，结果更亮"""
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    # Screen: result = 255 - ((255 - color1) * (255 - color2)) / 255
    blended = tuple(
        int(255 - ((255 - rgb1[i]) * (255 - rgb2[i])) / 255)
        for i in range(3)
    )
    return rgb_to_hex(blended)

def blend_colors_overlay(color1, color2):
    """使用Overlay混合模式：结合multiply和screen，适合双变量地图"""
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    blended = []
    for i in range(3):
        # 标准的overlay模式：基于base color (rgb1)决定使用multiply还是screen
        if rgb1[i] < 128:
            # 暗色区域：2 * multiply
            result = int(2 * (rgb1[i] * rgb2[i]) / 255)
        else:
            # 亮色区域：2 * screen - 1，但需要限制在0-255范围内
            screen_result = 255 - ((255 - rgb1[i]) * (255 - rgb2[i])) / 255
            result = int(2 * screen_result - 255)
        blended.append(max(0, min(255, result)))
    return rgb_to_hex(tuple(blended))

def blend_colors_soft_light(color1, color2):
    """使用Soft Light混合模式：柔和的混合"""
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    blended = []
    for i in range(3):
        if rgb2[i] < 128:
            result = int(rgb1[i] * (rgb2[i] / 128) * 0.5 + rgb1[i] * 0.5)
        else:
            result = int(rgb1[i] + (255 - rgb1[i]) * ((rgb2[i] - 128) / 128) * 0.5)
        blended.append(max(0, min(255, result)))
    return rgb_to_hex(tuple(blended))

# 默认使用multiply混合模式，适合从白色开始的双变量地图
BLEND_MODE = 'multiply'  # 可选: 'multiply', 'screen', 'overlay', 'soft_light'

def blend_colors(color1, color2, mode=BLEND_MODE):
    """根据指定模式混合两种颜色"""
    if mode == 'multiply':
        return blend_colors_multiply(color1, color2)
    elif mode == 'screen':
        return blend_colors_screen(color1, color2)
    elif mode == 'overlay':
        return blend_colors_overlay(color1, color2)
    elif mode == 'soft_light':
        return blend_colors_soft_light(color1, color2)
    else:
        # 默认使用multiply
        return blend_colors_multiply(color1, color2)

def get_color_by_custom_interval(value, intervals, colors):
    """根据值所在的自定义区间返回对应的颜色"""
    if value is None or pd.isna(value):
        return '#CCCCCC'  # 灰色表示缺失数据
    
    # 遍历区间，找到值所在的区间
    for i in range(len(intervals) - 1):
        if i == 0:
            # 第一个区间包含左边界
            if intervals[i] <= value < intervals[i + 1]:
                return colors[i]
        elif i == len(intervals) - 2:
            # 最后一个区间包含右边界
            if intervals[i] <= value <= intervals[i + 1]:
                return colors[i]
        else:
            # 中间区间
            if intervals[i] <= value < intervals[i + 1]:
                return colors[i]
    
    # 如果值超出范围
    if value < intervals[0]:
        return colors[0]
    elif value > intervals[-1]:
        return colors[-1]
    
    return '#CCCCCC'  # 默认灰色

def get_color_for_value(value, intervals, colors, variable_type='dpl'):
    """根据自定义区间获取颜色，用于双变量映射
    注意：值越小，颜色越淡（接近白色）；值越大，颜色越深
    这样在multiply混合模式下，两个小值混合会得到更浅的颜色
    """
    if value is None or pd.isna(value):
        return '#FFFFFF'  # 白色表示缺失，不影响混合
    
    # 使用自定义区间映射
    # 区间顺序：intervals[0]到intervals[-1]从小到大
    # colors[0]对应最小值区间（应该是最浅色），colors[-1]对应最大值区间（应该是最深色）
    return get_color_by_custom_interval(value, intervals, colors)

# 获取数据范围
min_val = None
max_val = None
min_val2 = None
max_val2 = None

if DATA_COLUMN in merged_gdf.columns:
    # 移除缺失值
    valid_data = merged_gdf.dropna(subset=[DATA_COLUMN])
    
    if not valid_data.empty:
        min_val = valid_data[DATA_COLUMN].min()
        max_val = valid_data[DATA_COLUMN].max()
        print(f"   - {DATA_COLUMN} 范围: {min_val:.2f} 到 {max_val:.2f}")
    else:
        print(f"   - 错误: 列 '{DATA_COLUMN}' 中没有有效数据")
else:
    print(f"   - 错误: 列 '{DATA_COLUMN}' 不存在于合并后的数据中")

# 如果使用双变量映射，获取第二个变量的数据范围
if USE_BIVARIATE and DATA_COLUMN2 in merged_gdf.columns:
    valid_data2 = merged_gdf.dropna(subset=[DATA_COLUMN2])
    if not valid_data2.empty:
        min_val2 = valid_data2[DATA_COLUMN2].min()
        max_val2 = valid_data2[DATA_COLUMN2].max()
        print(f"   - {DATA_COLUMN2} 范围: {min_val2:.2f} 到 {max_val2:.2f}")
    else:
        print(f"   - 错误: 列 '{DATA_COLUMN2}' 中没有有效数据")

# 定义样式函数
def style_function(feature):
    value1 = feature['properties'].get(DATA_COLUMN)
    value2 = feature['properties'].get(DATA_COLUMN2) if USE_BIVARIATE and DATA_COLUMN2 in merged_gdf.columns else None
    
    # 如果两个值都缺失
    if (value1 is None or pd.isna(value1)) and (value2 is None or pd.isna(value2) or not USE_BIVARIATE):
        return {
            'fillColor': '#CCCCCC',  # 灰色表示缺失数据
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5
        }
    
    # 单变量模式
    if not USE_BIVARIATE or min_val2 is None or max_val2 is None or value2 is None or pd.isna(value2):
        if value1 is None or pd.isna(value1):
            return {
                'fillColor': '#CCCCCC',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.5
            }
        return {
            'fillColor': get_color_for_value(value1, DPL_INTERVALS, DPL_COLORS, variable_type='dpl'),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 1
        }
    
    # 双变量模式：混合颜色
    if value1 is None or pd.isna(value1):
        # 如果第一个值缺失，只使用第二个值
        return {
            'fillColor': get_color_for_value(value2, DPE_INTERVALS, DPE_COLORS, variable_type='dpe'),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 1
        }
    
    # 获取两个变量的颜色（使用自定义区间）
    color1 = get_color_for_value(value1, DPL_INTERVALS, DPL_COLORS, variable_type='dpl')  # Dpl：橙色系
    color2 = get_color_for_value(value2, DPE_INTERVALS, DPE_COLORS, variable_type='dpe')  # Dpe：蓝色系
    
    # 使用multiply混合模式：适合从浅色开始的双变量地图
    blended_color = blend_colors(color1, color2, mode=BLEND_MODE)
    
    return {
        'fillColor': blended_color,
        'color': 'black',
        'weight': 1,
        'fillOpacity': 1
    }

style_function = style_function if min_val is not None and max_val is not None else None

# 创建双变量图例
def create_bivariate_legend(dpl_intervals, dpl_colors, dpe_intervals, dpe_colors, var1_name, var2_name, blend_mode='multiply'):
    """创建双变量图例（5x5网格）"""
    # 从变量名中提取"Dpl"和"Dpe"部分
    dpl_display_name = 'Dpl' if 'Dpl' in str(var1_name) else str(var1_name)
    dpe_display_name = 'Dpe' if 'Dpe' in str(var2_name) else str(var2_name)
    
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 420px; height: auto; 
                background-color: white; border: 2px solid black; 
                z-index:9999; font-size: 12px; padding: 8px; box-shadow: 3px 3px 10px rgba(0,0,0,0.3);">
        <h4 style="margin-top: 0; text-align: center; margin-bottom: 8px;">Bivariate Legend</h4>
        <table style="width: 100%; border-collapse: collapse; table-layout: fixed;">
            <tr>
                <td style="width: 60px;"></td>
                <td style="width: 60px;"></td>
                <td style="text-align: center; font-weight: bold; padding: 3px;" colspan="''' + str(len(dpe_intervals) - 1) + '''">''' + dpe_display_name + '''</td>
            </tr>
            <tr>
                <td style="width: 60px;"></td>
                <td style="width: 60px;"></td>
    '''
    
    # 添加Dpe区间的列标题（从左到右：从小到大）
    # 计算正方形单元格大小：确保5x5网格为正方形
    # 表格总宽度420px，减去padding 16px，减去左侧两列120px，剩余空间给5个正方形单元格
    table_width = 420 - 16  # 减去左右padding (8px * 2)
    remaining_width = table_width - 60 - 60  # 减去两个固定宽度列
    cell_size_px = remaining_width / (len(dpe_intervals) - 1)  # 每个单元格的宽度
    
    for i in range(0, len(dpe_intervals) - 1):
        if i == 0:
            interval_label = f"[{dpe_intervals[i]:.1f}, {dpe_intervals[i + 1]:.1f})"
        elif i == len(dpe_intervals) - 2:
            interval_label = f"[{dpe_intervals[i]:.1f}, {dpe_intervals[i + 1]:.1f}]"
        else:
            interval_label = f"[{dpe_intervals[i]:.1f}, {dpe_intervals[i + 1]:.1f})"
        legend_html += f'<td style="text-align: center; font-weight: bold; font-size: 10px; padding: 2px; width: {cell_size_px}px;">{interval_label}</td>'
    
    legend_html += '</tr>'
    
    # 创建5x5网格（从下到上：从小到大）
    for i in range(0, len(dpl_intervals) - 1):
        legend_html += '<tr>'
        # 在第一行（最小值的行）添加行标题
        if i == 0:
            legend_html += f'<td rowspan="{len(dpl_intervals)-1}" style="width: 60px; writing-mode: vertical-rl; text-align: center; font-weight: bold; vertical-align: middle; padding: 5px;">{dpl_display_name}</td>'
        
        # 添加Dpl区间标签（从下到上：从小到大）
        if i == 0:
            interval_label = f"[{dpl_intervals[i]:.1f}, {dpl_intervals[i + 1]:.1f})"
        elif i == len(dpl_intervals) - 2:
            interval_label = f"[{dpl_intervals[i]:.1f}, {dpl_intervals[i + 1]:.1f}]"
        else:
            interval_label = f"[{dpl_intervals[i]:.1f}, {dpl_intervals[i + 1]:.1f})"
        legend_html += f'<td style="width: 60px; text-align: center; font-weight: bold; padding: 3px; font-size: 10px;">{interval_label}</td>'
        
        # 为每个Dpe区间创建混合颜色（从左到右：从小到大）
        for j in range(0, len(dpe_intervals) - 1):
            # 计算区间中点值
            val1 = (dpl_intervals[i] + dpl_intervals[i + 1]) / 2
            val2 = (dpe_intervals[j] + dpe_intervals[j + 1]) / 2
            
            color1 = get_color_for_value(val1, dpl_intervals, dpl_colors, variable_type='dpl')
            color2 = get_color_for_value(val2, dpe_intervals, dpe_colors, variable_type='dpe')
            
            # 使用混合模式混合颜色
            blended = blend_colors(color1, color2, mode=blend_mode)
            
            # 使用正方形单元格确保5x5网格为正方形
            legend_html += f'<td style="background-color: {blended}; border: 1px solid black; width: {cell_size_px}px; height: {cell_size_px}px;"></td>'
        
        legend_html += '</tr>'
    
    # # 添加区间说明（从小到大顺序）
    # dpl_labels = []
    # for i in range(0, len(dpl_intervals) - 1):
    #     if i == 0:
    #         interval_label = f"[{dpl_intervals[i]:.1f}, {dpl_intervals[i + 1]:.1f})"
    #     elif i == len(dpl_intervals) - 2:
    #         interval_label = f"[{dpl_intervals[i]:.1f}, {dpl_intervals[i + 1]:.1f}]"
    #     else:
    #         interval_label = f"[{dpl_intervals[i]:.1f}, {dpl_intervals[i + 1]:.1f})"
    #     dpl_labels.append(f"Dpl{i+1}({interval_label})")
    
    # dpe_labels = []
    # for i in range(0, len(dpe_intervals) - 1):
    #     if i == 0:
    #         interval_label = f"[{dpe_intervals[i]:.1f}, {dpe_intervals[i + 1]:.1f})"
    #     elif i == len(dpe_intervals) - 2:
    #         interval_label = f"[{dpe_intervals[i]:.1f}, {dpe_intervals[i + 1]:.1f}]"
    #     else:
    #         interval_label = f"[{dpe_intervals[i]:.1f}, {dpe_intervals[i + 1]:.1f})"
    #     dpe_labels.append(f"Dpe{i+1}({interval_label})")
    
    legend_html += f'''
        </table>
        </div>
    </div>
    '''
    return legend_html

# 如果使用双变量映射，添加双变量图例
if USE_BIVARIATE and min_val is not None and max_val is not None and min_val2 is not None and max_val2 is not None:
    legend_html = create_bivariate_legend(DPL_INTERVALS, DPL_COLORS, DPE_INTERVALS, DPE_COLORS, DATA_COLUMN, DATA_COLUMN2, BLEND_MODE)
    m.get_root().html.add_child(folium.Element(legend_html))
    print(f"   - 已添加双变量图例（使用{BLEND_MODE}混合模式，5x5网格）")

# 添加GeoJSON图层
if style_function:
    folium.GeoJson(
        merged_gdf,
        name=f'{DATA_COLUMN}可视化' + (f' & {DATA_COLUMN2}双变量混合' if USE_BIVARIATE and DATA_COLUMN2 in merged_gdf.columns else ''),
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['地名', DATA_COLUMN] + ([DATA_COLUMN2] if USE_BIVARIATE and DATA_COLUMN2 in merged_gdf.columns else []),
            aliases=['城市:', f'{DATA_COLUMN}:'] + ([f'{DATA_COLUMN2}:'] if USE_BIVARIATE and DATA_COLUMN2 in merged_gdf.columns else []),
            localize=True,
            sticky=True,
            labels=True,
            style=(
                "background-color: #F0EFEF;"
                "border: 1px solid black;"
                "border-radius: 3px;"
                "box-shadow: 3px;"
                "font-size: 14px;"
            )
        ),
        popup=folium.GeoJsonPopup(
            fields=['地名', DATA_COLUMN] + ([DATA_COLUMN2] if USE_BIVARIATE and DATA_COLUMN2 in merged_gdf.columns else []),
            aliases=['城市:', f'{DATA_COLUMN}:'] + ([f'{DATA_COLUMN2}:'] if USE_BIVARIATE and DATA_COLUMN2 in merged_gdf.columns else []),
            localize=True,
            labels=True,
            style="width: 300px; font-size: 14px;"
        )
    ).add_to(m)
else:
    # 如果没有数据，只添加边界
    folium.GeoJson(
        merged_gdf,
        name='行政边界',
        style_function=lambda feature: {
            'fillColor': '#CCCCCC',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.3
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['地名'],
            aliases=['城市:'],
            localize=True,
            sticky=True
        )
    ).add_to(m)

# 添加地图控件
folium.LayerControl().add_to(m)  # 图层控制
folium.plugins.MousePosition().add_to(m)  # 鼠标位置
folium.plugins.Fullscreen(position='topright').add_to(m)  # 全屏按钮
folium.plugins.MeasureControl(position='bottomleft').add_to(m)  # 测量工具

# 添加标题
title_text = f'长三角城市{DATA_COLUMN}分布图(1990年)'
legend_note = ''
if USE_BIVARIATE and DATA_COLUMN2 in merged_gdf.columns:
    title_text = f'The Distribution Map of {DATA_COLUMN} and {DATA_COLUMN2} in the Yangtze River Delta (1990)'
    
title_html = f'''
    <h3 align="center" style="font-size:22px; margin-bottom:5px;">
        <b>{title_text}</b>
    </h3>
    {legend_note}
    <p align="center" style="font-size:16px; margin-top:0;">
        Source: The Research on the Structural Deviation of Population-Land-Economy in the Yangtze River Delta | Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
    </p>
'''
m.get_root().html.add_child(folium.Element(title_html))

# 5. 保存并显示地图
print("\n[步骤5/6] 保存地图...")
m.save(OUTPUT_HTML)
print(f"   - 地图已成功保存为: {OUTPUT_HTML}")

# 6. 导出高清PNG图片（无底图，仅行政边界和颜色填充）
print("\n[步骤6/6] 导出高清PNG图片...")

def get_province(city_name):
    """根据城市名称识别省份"""
    if pd.isna(city_name):
        return None
    city_name = str(city_name)
    # 上海市
    if '上海' in city_name:
        return '上海'
    # 江苏省
    elif any(city in city_name for city in ['南京', '苏州', '无锡', '常州', '镇江', '南通', '扬州', 
                                            '泰州', '盐城', '淮安', '宿迁', '徐州', '连云港']):
        return '江苏'
    # 浙江省
    elif any(city in city_name for city in ['杭州', '宁波', '温州', '嘉兴', '湖州', '绍兴', '金华', 
                                            '衢州', '舟山', '台州', '丽水']):
        return '浙江'
    # 安徽省
    elif any(city in city_name for city in ['合肥', '芜湖', '蚌埠', '淮南', '马鞍山', '淮北', '铜陵', 
                                            '安庆', '黄山', '滁州', '阜阳', '宿州', '六安', '亳州', 
                                            '池州', '宣城']):
        return '安徽'
    return None

def export_bivariate_high_res_map(merged_gdf, png_path, data_column1, data_column2, 
                                   dpl_intervals, dpl_colors, dpe_intervals, dpe_colors, blend_mode='multiply'):
    """使用matplotlib直接绘制高清双变量地图（无底图）"""
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 使用ASCII减号而不是Unicode减号
        # 确保负号正确显示
        import matplotlib
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        # 创建高分辨率图形
        fig, ax = plt.subplots(figsize=(16, 12), dpi=300)  # 高DPI获得高清图片
        
        # 获取双变量颜色
        def get_bivariate_color_png(val1, val2):
            """根据两个值获取双变量混合颜色"""
            if pd.isna(val1) or pd.isna(val2):
                return '#CCCCCC'  # 灰色表示缺失数据
            
            # 获取两个变量的颜色
            color1 = get_color_by_custom_interval(val1, dpl_intervals, dpl_colors)
            color2 = get_color_by_custom_interval(val2, dpe_intervals, dpe_colors)
            
            # 混合颜色
            return blend_colors(color1, color2, mode=blend_mode)
        
        # 为每个区域分配颜色
        merged_gdf['plot_color'] = merged_gdf.apply(
            lambda row: get_bivariate_color_png(
                row.get(data_column1), 
                row.get(data_column2)
            ),
            axis=1
        )
        
        # 添加省份信息
        merged_gdf['province'] = merged_gdf['地名'].apply(get_province)
        
        # 绘制地图（无底图，仅行政边界和颜色填充）
        merged_gdf.plot(
            ax=ax,
            color=merged_gdf['plot_color'],
            edgecolor='black',
            linewidth=0.5,
            legend=False
        )
        
        # 合并同一省份的城市，绘制加粗的省边界
        if 'province' in merged_gdf.columns:
            # 过滤掉省份为None的行
            province_gdf = merged_gdf[merged_gdf['province'].notna()].copy()
            if not province_gdf.empty:
                province_bounds = province_gdf.dissolve(by='province')
                province_bounds.plot(
                    ax=ax,
                    facecolor='none',
                    edgecolor='black',
                    linewidth=2.5,  # 加粗省边界
                    linestyle='-',
                    zorder=100  # 确保省边界在最上层
                )
        
        # 设置标题
        ax.set_title(
            f'The Distribution Map of {data_column1} and {data_column2} in the Yangtze River Delta',
            fontsize=20,
            fontweight='bold',
            pad=20
        )
        
        # 移除坐标轴
        ax.set_axis_off()
        
        # 设置坐标轴范围，使用统一边界确保三个地图对齐
        ax.set_xlim(UNIFIED_X_MIN, UNIFIED_X_MAX)
        ax.set_ylim(UNIFIED_Y_MIN, UNIFIED_Y_MAX)
        
        # 获取地图边界（用于图例位置计算）
        map_width = UNIFIED_X_MAX - UNIFIED_X_MIN
        map_height = UNIFIED_Y_MAX - UNIFIED_Y_MIN
        
        # 添加城市简称标签和象限分类标签
        if '简称' in merged_gdf.columns:
            for idx, row in merged_gdf.iterrows():
                if pd.notna(row.get('简称')) and pd.notna(row.get('geometry')):
                    try:
                        # 获取几何中心点
                        centroid = row.geometry.centroid
                        
                        # 计算象限分类（需要Dpl、Dpe、Dle）
                        dpl = row.get(data_column1)
                        dpe = row.get(data_column2)
                        dle = dpl - dpe if pd.notna(dpl) and pd.notna(dpe) else None
                        
                        # 添加城市简称标签
                        city_name = str(row['简称'])
                        ax.text(
                            centroid.x, centroid.y,
                            city_name,
                            fontsize=20,
                            ha='center',
                            va='center',
                            weight='bold',
                            color='black',
                            zorder=500
                        )
                        
                        # 添加象限分类标签（在城市名称下方）
                        if pd.notna(dpl) and pd.notna(dpe) and pd.notna(dle):
                            quadrant = classify_quadrant(dpl, dpe, dle)
                            # 提取象限编号（从"象限1-1"中提取"1-1"）
                            quadrant_num = quadrant.replace("象限", "") if quadrant != "未分类" else "?"
                            
                            # 在城市名称下方显示象限编号
                            ax.text(
                                centroid.x, centroid.y - 0.15,  # 在城市名称下方
                                quadrant_num,
                                fontsize=14,
                                ha='center',
                                va='center',
                                weight='bold',
                                color='black',  # 使用红色突出显示
                                zorder=501,
                                # bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', linewidth=1, alpha=0.5)
                            )
                    except Exception as e:
                        pass

        # 添加双变量图例（在地图右下角内部，不扩大范围）
        # 计算图例总宽度和高度
        # 交换：竖向为Dpe，横向为Dpl
        num_cols = len(dpl_intervals) - 1  # 横向：Dpl
        num_rows = len(dpe_intervals) - 1  # 竖向：Dpe
        
        # 获取figure的实际显示尺寸（英寸）
        # 这比使用transform更可靠，因为它在保存时不会变化
        fig_width_inches, fig_height_inches = fig.get_size_inches()
        
        # 计算axes在figure中的位置和大小
        # 获取axes的bbox（相对于figure的坐标）
        ax_bbox = ax.get_position()
        ax_width_inches = ax_bbox.width * fig_width_inches
        ax_height_inches = ax_bbox.height * fig_height_inches
        
        # 转换为像素（基于fig.dpi）
        ax_width_px = ax_width_inches * fig.dpi
        ax_height_px = ax_height_inches * fig.dpi
        
        # 计算每个数据单位对应的显示像素数
        x_pixels_per_unit = ax_width_px / map_width if map_width > 0 else 1
        y_pixels_per_unit = ax_height_px / map_height if map_height > 0 else 1
        
        # 确定正方形的显示像素大小（使用较小的比例，确保不占用太多空间）
        # 计算图例区域的可用显示像素
        legend_available_width_px = ax_width_px / 28  # 使用axes宽度的约1/28（增大图例）
        legend_available_height_px = ax_height_px / 28  # 使用axes高度的约1/28（增大图例）
        
        # 使用较小的值，确保图例不占用太多空间
        display_cell_size_px = min(legend_available_width_px, legend_available_height_px)
        
        # 将显示像素转换回数据坐标
        # 关键：为了让矩形在显示时是正方形，x和y方向需要使用不同的数据坐标值
        # 这样绘制出来的矩形在显示时会是正方形
        cell_size_x = display_cell_size_px / x_pixels_per_unit if x_pixels_per_unit > 0 else map_width / 28
        cell_size_y = display_cell_size_px / y_pixels_per_unit if y_pixels_per_unit > 0 else map_height / 28
        
        # 计算图例总尺寸（使用对应的x和y方向的值）
        legend_total_width = num_cols * cell_size_x
        legend_total_height = num_rows * cell_size_y
        
        # 图例起始位置（左下角，在地图内部）
        legend_x_start = UNIFIED_X_MIN + map_width * 0.02  # 图例在左侧，留出边距
        legend_y_start = UNIFIED_Y_MIN + map_height * 0.02  # 图例在底部，留出边距
        
        # 绘制Dpl边界数值标签（横轴，在网格边界竖线处）
        # 只标注边界值：-4, -2, 0, 2, 4, 6
        # 确保标签在坐标轴范围内
        dpl_label_y = max(UNIFIED_Y_MIN, legend_y_start - map_height * 0.01)
        for j in range(len(dpl_intervals)):
            x_pos = legend_x_start + j * cell_size_x
            # 确保x_pos在范围内
            x_pos = max(UNIFIED_X_MIN, min(UNIFIED_X_MAX, x_pos))
            # 确保负号正确显示（直接格式化，避免Unicode问题）
            val = dpl_intervals[j]
            if val == int(val):
                # 整数：直接使用字符串格式化，确保负号显示
                label_text = f'{val:+d}'.replace('+', '')  # 使用+格式然后去掉正号
            else:
                label_text = f'{val:.1f}'
            # 确保使用ASCII减号
            label_text = label_text.replace('−', '-').replace('–', '-')
            ax.text(
                x_pos, dpl_label_y+0.05,
                label_text,
                fontsize=12,
                ha='center',
                va='top',
                weight='bold',
                clip_on=False  # 关闭裁剪，确保标签完整显示
            )
        
        # 添加Dpl变量名标签（在数值标签下方，但确保在范围内）
        # 从变量名中提取"Dpl"部分（例如从"Dpl2020"提取"Dpl"）
        dpl_var_name = 'Dpl' if 'Dpl' in str(data_column1) else str(data_column1)
        dpl_var_y = max(UNIFIED_Y_MIN, dpl_label_y - map_height * 0.02)
        ax.text(
            legend_x_start + legend_total_width / 2,
            dpl_var_y,
            dpl_var_name,
            fontsize=15,
            ha='center',
            va='top',
            weight='bold',
            clip_on=False  # 关闭裁剪，确保标签完整显示
        )
        
        # 绘制Dpe边界数值标签（纵轴，在网格边界横线处）
        # 只标注边界值：-15, -4, -2, 0, 2, 4
        # 负数在下方，正数在上方（从下到上：-15, -4, -2, 0, 2, 4）
        # 确保标签在坐标轴范围内
        dpe_label_x = max(UNIFIED_X_MIN, legend_x_start - map_width * 0.01)
        for i in range(len(dpe_intervals)):
            # 从下到上标注：i=0对应最底部（-15），i=len(dpe_intervals)-1对应最顶部（4）
            # 使用 i 使负数在下方，正数在上方
            y_pos = legend_y_start + i * cell_size_y
            # 确保y_pos在范围内
            y_pos = max(UNIFIED_Y_MIN, min(UNIFIED_Y_MAX, y_pos))
            # 确保负号正确显示（直接格式化，避免Unicode问题）
            val = dpe_intervals[i]
            if val == int(val):
                # 整数：直接使用字符串格式化，确保负号显示
                label_text = f'{val:+d}'.replace('+', '')  # 使用+格式然后去掉正号
            else:
                label_text = f'{val:.1f}'
            # 确保使用ASCII减号
            label_text = label_text.replace('−', '-').replace('–', '-')
            ax.text(
                dpe_label_x, y_pos+0.05,
                label_text,
                fontsize=12,
                ha='right',
                va='center',
                weight='bold',
                clip_on=False  # 关闭裁剪，确保标签完整显示
            )
        
        # 添加Dpe变量名标签（垂直，在数值标签左侧，但确保在范围内）
        # 从变量名中提取"Dpe"部分（例如从"Dpe2020"提取"Dpe"）
        dpe_var_name = 'Dpe' if 'Dpe' in str(data_column2) else str(data_column2)
        dpe_var_x = max(UNIFIED_X_MIN, dpe_label_x - map_width * 0.02)
        ax.text(
            dpe_var_x-0.1,
            legend_y_start + legend_total_height / 2,
            dpe_var_name,
            fontsize=15,
            ha='right',
            va='center',
            weight='bold',
            rotation=90,
            clip_on=False  # 关闭裁剪，确保标签完整显示
        )
        
        # 绘制颜色网格（从下到上，从左到右，无间距，确保显示时是正方形）
        # 竖向为Dpe（从下到上：-15, -4, -2, 0, 2, 4），横向为Dpl（从左到右：-4, -2, 0, 2, 4, 6）
        for i in range(num_rows):  # i循环Dpe（竖向）
            for j in range(num_cols):  # j循环Dpl（横向）
                # 计算区间中点值（注意：i对应Dpe，j对应Dpl）
                val1 = (dpe_intervals[i] + dpe_intervals[i + 1]) / 2  # Dpe（竖向）
                val2 = (dpl_intervals[j] + dpl_intervals[j + 1]) / 2  # Dpl（横向）
                
                # 获取混合颜色
                color1 = get_color_by_custom_interval(val1, dpe_intervals, dpe_colors)  # Dpe颜色
                color2 = get_color_by_custom_interval(val2, dpl_intervals, dpl_colors)  # Dpl颜色
                blended_color = blend_colors(color1, color2, mode=blend_mode)
                
                # 计算位置（从下到上，从左到右，无间距）
                # i对应竖向（Dpe），j对应横向（Dpl）
                x_pos = legend_x_start + j * cell_size_x  # 横向：Dpl
                y_pos = legend_y_start + i * cell_size_y  # 竖向：Dpe（负数在下方）
                
                # 绘制颜色方块（在显示时是正方形，使用不同的x和y数据坐标值）
                from matplotlib.patches import Rectangle
                rect = Rectangle(
                    (x_pos, y_pos),
                    cell_size_x,  # x方向的宽度
                    cell_size_y,  # y方向的高度
                    facecolor=blended_color,
                    edgecolor='black',
                    linewidth=1,
                    zorder=1000
                )
                ax.add_patch(rect)
        
        # 添加图例标题（在网格上方，但确保在范围内）
        legend_title_y = min(UNIFIED_Y_MAX, legend_y_start + legend_total_height + map_height * 0.008)
        ax.text(
            legend_x_start + legend_total_width / 2,
            legend_title_y,
            'Bivariate Legend',
            fontsize=15,
            ha='center',
            va='bottom',
            weight='bold',
            clip_on=True
        )
        
        # 再次确保坐标轴范围固定（防止图例影响）
        ax.set_xlim(UNIFIED_X_MIN, UNIFIED_X_MAX)
        ax.set_ylim(UNIFIED_Y_MIN, UNIFIED_Y_MAX)
        
        # 使用固定的坐标轴位置，确保所有地图布局完全一致
        # [left, bottom, width, height] 相对于figure的比例
        ax.set_position([0.05, 0.05, 0.90, 0.90])
        
        # 保存高清PNG（使用固定边距，确保图像尺寸一致）
        print(f"   - 正在保存高清PNG图片: {png_path}")
        plt.savefig(
            png_path,
            dpi=300,
            bbox_inches=None,  # 使用None，配合subplots_adjust确保图像尺寸一致
            pad_inches=0,  # 无额外边距
            facecolor='white',
            edgecolor='none',
            format='png'
        )
        
        plt.close()
        
        print(f"   - 高清PNG图片已成功保存为: {png_path}")
        return True
        
    except Exception as e:
        print(f"   - 导出PNG失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# 导出高清PNG（双变量地图）
if USE_BIVARIATE and DATA_COLUMN in merged_gdf.columns and DATA_COLUMN2 in merged_gdf.columns:
    if min_val is not None and max_val is not None and min_val2 is not None and max_val2 is not None:
        export_bivariate_high_res_map(
            merged_gdf, 
            OUTPUT_PNG, 
            DATA_COLUMN, 
            DATA_COLUMN2,
            DPL_INTERVALS,
            DPL_COLORS,
            DPE_INTERVALS,
            DPE_COLORS,
            BLEND_MODE
        )
    else:
        print(f"   - 警告: 数据列缺失值，跳过PNG导出")
else:
    print(f"   - 警告: 未启用双变量映射或数据列不存在，跳过PNG导出")

# 在浏览器中打开地图
try:
    webbrowser.open(OUTPUT_HTML)
    print("   - 正在在默认浏览器中打开地图...")
except:
    print("   - 无法自动打开浏览器，请手动打开生成的地图文件")

print("\n" + "="*80)
print("处理完成！")
print("="*80)