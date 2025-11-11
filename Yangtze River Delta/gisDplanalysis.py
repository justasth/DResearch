import folium
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from shapely.geometry import Point
import webbrowser
from folium import plugins
from branca.colormap import LinearColormap
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.patches import Circle, Polygon
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

# 配置参数
SHP_PATH = "data/长三角市级边界.shp"
EXCEL_PATH  = "analysisdata.xlsx"
OUTPUT_HTML = "长三角人地结构偏离度地图_2020.html"
OUTPUT_PNG = OUTPUT_HTML.replace('.html', '.png')  # PNG输出文件名
DATA_COLUMN = "Dpl2020"  # 要可视化的数据列

# 统一的坐标轴范围（确保三个地图对齐）
UNIFIED_X_MIN = 114.878463
UNIFIED_Y_MIN = 27.143423
UNIFIED_X_MAX = 122.834203
UNIFIED_Y_MAX = 35.127197

# 1. 加载行政边界数据
print("\n[步骤1/6] 加载行政边界数据...")
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
print("\n[步骤2/6] 加载Excel数据...")
df = pd.read_excel(EXCEL_PATH)
print(f"   - Excel数据加载完成，包含 {len(df)} 行")
print(f"   - 列名: {df.columns.tolist()}")

# 3. 合并地理数据与Excel数据
print("\n[步骤3/6] 合并地理数据和Excel数据...")
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
print("\n[步骤4/6] 创建交互式地图...")

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

# 创建颜色映射
# 初始化变量，确保在导出PNG时可用
min_val = None
max_val = None
colors = None

if DATA_COLUMN in merged_gdf.columns:
    # 移除缺失值
    valid_data = merged_gdf.dropna(subset=[DATA_COLUMN])
    
    if not valid_data.empty:
        min_val = -4#valid_data[DATA_COLUMN].min()
        max_val = 6#valid_data[DATA_COLUMN].max()
        
        # 定义颜色范围：从浅橙色到深橙色（5个等级）
        colors = ['#FFE5CC', '#FFCC99', '#FF9900', '#CC6600', '#993300']
        
        # 创建区间等级颜色映射函数
        def get_color_by_interval(value, min_val, max_val, colors):  
            """根据值所在的区间返回对应的颜色（渐变色）"""
            if value is None or pd.isna(value):
                return '#CCCCCC'  # 灰色表示缺失数据
            
            # 计算区间大小
            interval_size = (max_val - min_val) / len(colors)
            
            # 确定值所在的区间（0到len(colors)-1）
            if value <= min_val:
                interval_index = 0
            elif value >= max_val:
                interval_index = len(colors) - 1
            else:
                interval_index = int((value - min_val) / interval_size)
                # 确保索引在有效范围内
                if interval_index >= len(colors):
                    interval_index = len(colors) - 1
            
            return colors[interval_index]
        
        # 创建区间等级图例函数
        def create_interval_legend(min_val, max_val, colors, column_name):
            """创建区间等级图例HTML"""
            interval_size = (max_val - min_val) / len(colors)
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; right: 50px; width: 180px; height: auto;
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:12px; padding: 10px; border-radius: 5px;
                        box-shadow: 0 0 15px rgba(0,0,0,0.2);">
                <p style="margin: 0 0 8px 0; font-weight: bold; font-size: 14px; text-align: center;">
                    ''' + column_name + '''<br>Legend
                </p>
                <table style="width: 100%; border-collapse: collapse;">
            '''
            
            # 从最高等级到最低等级显示（从上到下）
            for i in range(len(colors) - 1, -1, -1):
                # 计算区间边界
                if i == 0:
                    interval_min = min_val
                    interval_max = min_val + interval_size
                    interval_label = f"[{interval_min:.1f}, {interval_max:.1f})"
                elif i == len(colors) - 1:
                    interval_min = min_val + i * interval_size
                    interval_max = max_val
                    interval_label = f"[{interval_min:.1f}, {interval_max:.1f}]"
                else:
                    interval_min = min_val + i * interval_size
                    interval_max = min_val + (i + 1) * interval_size
                    interval_label = f"[{interval_min:.1f}, {interval_max:.1f})"
                
                legend_html += f'''
                    <tr>
                        <td style="width: 30px; height: 25px; background-color: {colors[i]}; 
                                   border: 1px solid black; padding: 0;"></td>
                        <td style="padding-left: 8px; vertical-align: middle; font-size: 11px;">
                            {interval_label}
                        </td>
                    </tr>
                '''
            
            legend_html += '''
                </table>
            </div>
            '''
            return legend_html
        
        # 添加区间等级图例
        legend_html = create_interval_legend(min_val, max_val, colors, DATA_COLUMN)
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # 定义样式函数
        def style_function(feature):
            value = feature['properties'].get(DATA_COLUMN)
            if value is None or pd.isna(value):
                return {
                    'fillColor': '#CCCCCC',  # 灰色表示缺失数据
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.5
                }
            # 使用区间等级颜色映射
            return {
                'fillColor': get_color_by_interval(value, min_val, max_val, colors),
                'color': 'black',
                'weight': 1,
                'fillOpacity': 1
            }
    else:
        print(f"   - 错误: 列 '{DATA_COLUMN}' 中没有有效数据")
        style_function = None
else:
    print(f"   - 错误: 列 '{DATA_COLUMN}' 不存在于合并后的数据中")
    style_function = None

# 添加GeoJSON图层
if style_function:
    folium.GeoJson(
        merged_gdf,
        name=f'{DATA_COLUMN}可视化',
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['地名', DATA_COLUMN],
            aliases=['城市:', f'{DATA_COLUMN}:'],
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
            fields=['地名', DATA_COLUMN],
            aliases=['城市:', f'{DATA_COLUMN}:'],
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
title_html = f'''
    <h3 align="center" style="font-size:22px; margin-bottom:5px;">
        <b>The Distribution Map of {DATA_COLUMN} in the Yangtze River Delta (1990)</b>
    </h3>
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

# def add_north_arrow(ax, x, y, length=0.03, width=0.01):
#     """添加指北针"""
#     # 计算指北针的位置（地图坐标）
#     # 绘制指北针箭头（指向北）
#     arrow = FancyArrowPatch(
#         (x, y),
#         (x, y + length),
#         arrowstyle='->', 
#         mutation_scale=20,
#         color='black',
#         linewidth=2,
#         zorder=1000
#     )
#     ax.add_patch(arrow)
    
    # # 添加N标记
    # ax.text(x, y + length * 1.2, 'N', 
    #         fontsize=14, fontweight='bold', 
    #         ha='center', va='bottom',
    #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))

# def add_scale_bar(ax, x, y, length_km=100, crs=None):
#     """添加比例尺"""
#     # 如果给出了CRS，计算长度对应的地图单位
#     # 这里假设是WGS84，大约1度=111km
#     if crs is None or crs.to_epsg() == 4326:
#         # WGS84坐标系
#         length_deg = length_km / 111.0
    
#     # 绘制比例尺基线
#     ax.plot([x, x + length_deg], [y, y], 
#             color='black', linewidth=3, zorder=1000)
    
#     # 绘制比例尺两端的竖线
#     ax.plot([x, x], [y - length_deg*0.02, y + length_deg*0.02], 
#             color='black', linewidth=3, zorder=1000)
#     ax.plot([x + length_deg, x + length_deg], [y - length_deg*0.02, y + length_deg*0.02], 
#             color='black', linewidth=3, zorder=1000)
    
#     # 添加标签
#     ax.text(x + length_deg/2, y - length_deg*0.05, 
#             f'{length_km} km', 
#             fontsize=10, fontweight='bold',
#             ha='center', va='top',
#             # bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8)
#             )

def export_high_res_map(merged_gdf, png_path, data_column, min_val, max_val, colors):
    """使用matplotlib直接绘制高清地图（无底图）"""
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建高分辨率图形
        fig, ax = plt.subplots(figsize=(16, 12), dpi=300)  # 高DPI获得高清图片
        
        # 准备颜色映射
        def get_color_by_interval(value, min_val, max_val, colors):
            """根据值所在的区间返回对应的颜色"""
            if value is None or pd.isna(value):
                return '#CCCCCC'  # 灰色表示缺失数据
            
            interval_size = (max_val - min_val) / len(colors)
            
            if value <= min_val:
                interval_index = 0
            elif value >= max_val:
                interval_index = len(colors) - 1
            else:
                interval_index = int((value - min_val) / interval_size)
                if interval_index >= len(colors):
                    interval_index = len(colors) - 1
            
            return colors[interval_index]
        
        # 为每个区域分配颜色
        merged_gdf['plot_color'] = merged_gdf[data_column].apply(
            lambda x: get_color_by_interval(x, min_val, max_val, colors)
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
            f'The Distribution Map of {data_column} in the Yangtze River Delta',
            fontsize=20,
            fontweight='bold',
            pad=20
        )
        
        # 设置坐标轴范围，使用统一边界确保三个地图对齐
        ax.set_xlim(UNIFIED_X_MIN, UNIFIED_X_MAX)
        ax.set_ylim(UNIFIED_Y_MIN, UNIFIED_Y_MAX)
        
        # 移除坐标轴
        ax.set_axis_off()
        
        # 创建图例
        interval_size = (max_val - min_val) / len(colors)
        legend_elements = []
        
        # 从高到低创建图例项
        for i in range(len(colors) - 1, -1, -1):
            if i == 0:
                interval_min = min_val
                interval_max = min_val + interval_size
                label = f'[{interval_min:.1f}, {interval_max:.1f})'
            elif i == len(colors) - 1:
                interval_min = min_val + i * interval_size
                interval_max = max_val
                label = f'[{interval_min:.1f}, {interval_max:.1f}]'
            else:
                interval_min = min_val + i * interval_size
                interval_max = min_val + (i + 1) * interval_size
                label = f'[{interval_min:.1f}, {interval_max:.1f})'
            
            legend_elements.append(Patch(facecolor=colors[i], edgecolor='black', label=label))
        
        # 添加缺失数据的图例项
        legend_elements.append(Patch(facecolor='#CCCCCC', edgecolor='black', label='Nodata'))

        # 添加图例
        ax.legend(
            handles=legend_elements,
            loc='lower left',
            fontsize=15,
            title=data_column,
            title_fontsize=20,
            # frameon=True,
            # fancybox=True,
            # shadow=True
        )
        
        # 添加城市简称标签
        if '简称' in merged_gdf.columns:
            for idx, row in merged_gdf.iterrows():
                if pd.notna(row.get('简称')) and pd.notna(row.get('geometry')):
                    try:
                        # 获取几何中心点
                        centroid = row.geometry.centroid
                        ax.text(
                            centroid.x, centroid.y,
                            str(row['简称']),
                            fontsize=20,
                            ha='center',
                            va='center',
                            weight='bold',
                            color='black',
                            # bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7),
                            zorder=500
                        )
                    except:
                        pass
        
        # # 获取地图边界用于定位指北针和比例尺
        # x_min, y_min, x_max, y_max = merged_gdf.total_bounds

        # # 添加指北针（右上角）
        # arrow_x = x_max - (x_max - x_min) * 0.08
        # arrow_y = x_max - (x_max - x_min) * 0.08
        # arrow_length = (x_max - x_min) * 0.03
        # add_north_arrow(ax, arrow_x, arrow_y, length=arrow_length)

        # # 添加比例尺（右上角）
        # scale_x = x_max - (x_max - x_min) * 0.1
        # scale_y = y_max - (y_max - y_min) * 0.1
        # add_scale_bar(ax, scale_x, scale_y, length_km=100, crs=merged_gdf.crs)
        
        # 添加数据来源信息
        # ax.text(
        #     0.02, 0.02,
        #     f'Source: The Research on the Structural Deviation of Population-Land-Economy in the Yangtze River Delta | Date: {pd.Timestamp.now().strftime("%Y-%m-%d")}',
        #     transform=ax.transAxes,
        #     fontsize=10,
        #     verticalalignment='bottom',
        #     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # )
        
        # 再次确保坐标轴范围固定
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
        
# 导出高清PNG
if min_val is not None and max_val is not None and colors is not None:
    export_high_res_map(merged_gdf, OUTPUT_PNG, DATA_COLUMN, min_val, max_val, colors)
else:
    print(f"   - 警告: 颜色映射参数未定义，跳过PNG导出")

# 在浏览器中打开地图
try:
    webbrowser.open(OUTPUT_HTML)
    print("   - 正在在默认浏览器中打开地图...")
except:
    print("   - 无法自动打开浏览器，请手动打开生成的地图文件")

print("\n" + "="*80)
print("处理完成！")
print("="*80)
