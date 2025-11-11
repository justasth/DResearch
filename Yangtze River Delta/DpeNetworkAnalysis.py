import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    print("警告: networkx未安装，将使用matplotlib直接绘制网络图")
    HAS_NETWORKX = False
import geopandas as gpd

# 配置参数
EXCEL_PATH = "analysisdata.xlsx"
SHP_PATH = "data/长三角市级边界.shp"
DATA_COLUMN = "Dle2020"
OUTPUT_PNG = "Dle2020网络分析图.png"
legend_label="Dle"

# 统一的坐标轴范围（确保与其他地图对齐）
UNIFIED_X_MIN = 114.878463
UNIFIED_Y_MIN = 27.143423
UNIFIED_X_MAX = 122.834203
UNIFIED_Y_MAX = 35.127197

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("\n[步骤1/4] 加载数据...")
# 加载Excel数据
df = pd.read_excel(EXCEL_PATH)
print(f"   - Excel数据加载完成，包含 {len(df)} 行")

# 加载地理数据（用于获取城市坐标）
gdf = gpd.read_file(SHP_PATH, engine='fiona', encoding='utf-8')
if gdf.crs is None or (hasattr(gdf.crs, 'to_epsg') and gdf.crs.to_epsg() != 4326):
    gdf = gdf.to_crs(epsg=4326)

# 合并数据
merged_df = gdf.merge(df, left_on='地名', right_on='全称', how='left')

# 过滤有效数据
valid_data = merged_df[merged_df[DATA_COLUMN].notna()].copy()
print(f"   - 有效数据: {len(valid_data)} 个城市")
print(f"   - 正值城市: {(valid_data[DATA_COLUMN] > 0).sum()} 个")
print(f"   - 负值城市: {(valid_data[DATA_COLUMN] < 0).sum()} 个")

print("\n[步骤2/4] 排序和配对...")
# 分离正值和负值
positive = valid_data[valid_data[DATA_COLUMN] > 0].copy()
negative = valid_data[valid_data[DATA_COLUMN] < 0].copy()

# 排序：正值从大到小，负值按绝对值从大到小（即从最负的开始）
positive = positive.sort_values(DATA_COLUMN, ascending=False).reset_index(drop=True)
negative = negative.sort_values(DATA_COLUMN, ascending=True).reset_index(drop=True)  # 从小到大即从最负的开始

print(f"   - 正值排序（从大到小）:")
for i, row in positive.head(5).iterrows():
    print(f"      {i+1}. {row['全称']}: {row[DATA_COLUMN]:.3f}")

print(f"   - 负值排序（从小到大，即从绝对值最大到最小）:")
for i, row in negative.head(5).iterrows():
    print(f"      {i+1}. {row['全称']}: {row[DATA_COLUMN]:.3f} (绝对值: {abs(row[DATA_COLUMN]):.3f})")

# 配对逻辑：正值第一分给负值第一（绝对值最大的负值），如果没有分完就分给负值第二，如果不够分就从正值第二名开始分
edges = []  # 存储边（配对关系）
edge_weights = {}  # 存储边的权重（分配的值）

# 计算总的正值和负值
total_positive = positive[DATA_COLUMN].sum()
total_negative = abs(negative[DATA_COLUMN].sum())

print(f"\n   - 正值总和: {total_positive:.3f}")
print(f"   - 负值总和（绝对值）: {total_negative:.3f}")

# 配对算法：正值第一分给负值第一（绝对值最大的负值）
positive_idx = 0  # 正值索引（从第一开始）
negative_idx = 0  # 负值索引（从第一开始，即绝对值最大的负值）

while positive_idx < len(positive) and negative_idx >= 0:
    pos_city = positive.iloc[positive_idx]
    neg_city = negative.iloc[negative_idx]
    
    pos_value = pos_city[DATA_COLUMN]
    neg_value = abs(neg_city[DATA_COLUMN])
    
    # 计算可以分配的值（取较小值）
    allocation = min(pos_value, neg_value)
    
    # 创建边
    edge = (pos_city['全称'], neg_city['全称'])
    edges.append(edge)
    edge_weights[edge] = allocation
    
    print(f"   - 配对: {pos_city['全称']} ({pos_value:.3f}) -> {neg_city['全称']} ({-neg_value:.3f}), 分配值: {allocation:.3f}")
    
    # 更新剩余值
    pos_value -= allocation
    neg_value -= allocation
    
    # 更新DataFrame中的值（用于后续配对）
    positive.at[positive_idx, DATA_COLUMN] = pos_value
    negative.at[negative_idx, DATA_COLUMN] = -neg_value  # 保持负值
    
    # 如果正值用完了，移动到下一个正值
    if pos_value < 1e-6:  # 使用小的阈值避免浮点误差
        positive_idx += 1
    
    # 如果负值用完了，移动到下一个负值（第二，即绝对值第二大的负值）
    if neg_value < 1e-6:
        negative_idx += 1

print(f"\n   - 总共形成 {len(edges)} 个配对关系")

print("\n[步骤3/4] 构建网络图...")
if HAS_NETWORKX:
    # 创建有向图
    G = nx.DiGraph()

    # 添加节点
    all_cities = pd.concat([positive, negative]).drop_duplicates(subset=['全称'])
    for idx, row in all_cities.iterrows():
        city_name = row['全称']
        dpe_value = row[DATA_COLUMN]
        G.add_node(city_name, dpe_value=dpe_value, 
                    is_positive=dpe_value > 0,
                    city_name=city_name)

    # 添加边
    for edge in edges:
        source, target = edge
        weight = edge_weights[edge]
        G.add_edge(source, target, weight=weight)

    # 计算出入度
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    print(f"   - 节点数: {G.number_of_nodes()}")
    print(f"   - 边数: {G.number_of_edges()}")
    print(f"   - 最大入度: {max(in_degree.values()) if in_degree else 0}")
    print(f"   - 最大出度: {max(out_degree.values()) if out_degree else 0}")
else:
    # 如果没有networkx，计算出入度
    all_cities = pd.concat([positive, negative]).drop_duplicates(subset=['全称'])
    in_degree = {}
    out_degree = {}
    for city in all_cities['全称']:
        in_degree[city] = sum(1 for s, t in edges if t == city)
        out_degree[city] = sum(1 for s, t in edges if s == city)

print("\n[步骤4/4] 绘制网络图...")
# 创建图形（与gisDpeanalysis.py保持一致）
fig, ax = plt.subplots(figsize=(16, 12), dpi=300)  # 高DPI获得高清图片

# 首先绘制地图边界
print("   - 绘制行政边界...")
# 绘制所有城市的边界
merged_df.plot(ax=ax, 
               facecolor='white',
               edgecolor='black',
               linewidth=0.5,
               alpha=0.3,
               zorder=0)

# 添加省份信息并绘制加粗的省边界
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

# 添加省份列
merged_df['province'] = merged_df['地名'].apply(get_province)

# 绘制加粗的省边界
province_gdf = merged_df[merged_df['province'].notna()].copy()
if not province_gdf.empty:
    province_bounds = province_gdf.dissolve(by='province')
    province_bounds.plot(ax=ax,
                        facecolor='none',
                        edgecolor='black',
                        linewidth=2.5,  # 加粗省边界
                        linestyle='-',
                        zorder=1)

# 获取城市坐标（使用行政边界的几何中心）
# 注意：配对过程中修改了DataFrame的值，需要使用原始值
pos = {}
city_data = {}
# 从valid_data获取原始Dpe值（配对前的值）
original_dpe_dict = valid_data.set_index('全称')[DATA_COLUMN].to_dict()

for idx, row in all_cities.iterrows():
    city_name = row['全称']
    if pd.notna(row.get('geometry')):
        centroid = row.geometry.centroid
        pos[city_name] = (centroid.x, centroid.y)
        # 使用原始Dpe值（配对前的值）
        original_dpe_value = original_dpe_dict.get(city_name, row[DATA_COLUMN])
        city_data[city_name] = {
            'dpe_value': original_dpe_value,
            'abs_value': abs(original_dpe_value),
            'is_positive': original_dpe_value > 0,
            'out_degree': out_degree.get(city_name, 0),
            'in_degree': in_degree.get(city_name, 0)
        }
    else:
        print(f"   - 警告: {city_name} 缺少地理信息")

# 计算节点大小范围（基于绝对值）
abs_values = [city_data[c]['abs_value'] for c in city_data]
min_abs = min(abs_values) if abs_values else 0.1
max_abs = max(abs_values) if abs_values else 1.0
print(f"   - 绝对值范围: {min_abs:.3f} 到 {max_abs:.3f}")

# 节点大小映射函数（基于绝对值）
def get_node_size(abs_value, min_val=min_abs, max_val=max_abs):
    """根据绝对值计算节点大小（返回数据坐标单位）"""
    # 映射到 0.1-0.5 度范围（适合地理坐标系统）
    # 长三角地区大约8度宽，所以0.1-0.5度是合理的节点大小
    size_deg = 0.1 + (abs_value - min_val) / (max_val - min_val) * 0.4
    return max(0.05, min(0.5, size_deg))

# 绘制边（连线表示合作，线的粗细表示分配值的大小）
print("   - 绘制合作连线...")
# 计算分配值的范围，用于归一化线宽
allocation_values = list(edge_weights.values())
min_allocation = min(allocation_values) if allocation_values else 0.01
max_allocation = max(allocation_values) if allocation_values else 1.0

# 定义线宽范围（像素）
MIN_LINE_WIDTH = 1  # 最小线宽
MAX_LINE_WIDTH = 10  # 最大线宽

print(f"   - 分配值范围: {min_allocation:.3f} 到 {max_allocation:.3f}")
print(f"   - 线宽范围: {MIN_LINE_WIDTH} 到 {MAX_LINE_WIDTH} 像素")

for edge in edges:
    source, target = edge
    if source in pos and target in pos:
        x1, y1 = pos[source]
        x2, y2 = pos[target]
        # 线的粗细基于分配值（edge_weights），映射到指定范围
        allocation = edge_weights.get(edge, 0)
        
        # 归一化到指定线宽范围
        if max_allocation > min_allocation:
            # 归一化到0-1范围
            normalized_allocation = (allocation - min_allocation) / (max_allocation - min_allocation)
            # 映射到线宽范围
            line_width = MIN_LINE_WIDTH + normalized_allocation * (MAX_LINE_WIDTH - MIN_LINE_WIDTH)
        else:
            # 如果所有分配值相同，使用中等线宽
            line_width = (MIN_LINE_WIDTH + MAX_LINE_WIDTH) / 2
        
        ax.plot([x1, x2], [y1, y2], 
               color='gray', 
               alpha=0.6,
               linewidth=line_width,
               zorder=2)

# 绘制节点（气泡图，大小基于Dpe绝对值，颜色表示正负）
print("   - 绘制城市节点（气泡图）...")

# 先设置坐标轴范围和位置
ax.set_xlim(UNIFIED_X_MIN, UNIFIED_X_MAX)
ax.set_ylim(UNIFIED_Y_MIN, UNIFIED_Y_MAX)
ax.set_position([0.05, 0.05, 0.90, 0.90])

# 计算坐标系统的纵横比，确保绘制正圆
# 获取figure和axes的尺寸（英寸）
fig_width_inch, fig_height_inch = fig.get_size_inches()
ax_pos = ax.get_position()
ax_width_inch = ax_pos.width * fig_width_inch
ax_height_inch = ax_pos.height * fig_height_inch

# 计算数据坐标范围
x_range = UNIFIED_X_MAX - UNIFIED_X_MIN
y_range = UNIFIED_Y_MAX - UNIFIED_Y_MIN

# 计算每度对应的像素数
dpi = fig.dpi
x_pixels_per_deg = (ax_width_inch * dpi) / x_range
y_pixels_per_deg = (ax_height_inch * dpi) / y_range

# 计算纵横比（y方向像素数 / x方向像素数）
aspect_ratio = y_pixels_per_deg / x_pixels_per_deg

for city_name, (x, y) in pos.items():
    if city_name in city_data:
        data = city_data[city_name]
        abs_value = data['abs_value']
        is_positive = data['is_positive']
        
        # 节点大小基于Dpe绝对值（数据坐标单位）
        node_size_data = get_node_size(abs_value)
        
        # 计算在显示坐标中的半径（像素）
        radius_pixels_x = node_size_data * x_pixels_per_deg
        radius_pixels_y = node_size_data * y_pixels_per_deg
        
        # 为了在显示时为正圆，取较小的半径（像素）
        radius_pixels = min(radius_pixels_x, radius_pixels_y)
        
        # 转换回数据坐标，分别计算x和y方向的半径
        # 使用Ellipse可以分别控制width和height
        # 计算在数据坐标中的半径
        # width和height需要根据aspect ratio调整
        width_data = radius_pixels / x_pixels_per_deg * 2  # Ellipse的width是直径
        height_data = radius_pixels / y_pixels_per_deg * 2  # Ellipse的height是直径
        
        # 颜色：正值用橙色系，负值用蓝色系
        if is_positive:
            node_color = '#FF9900'  
        else:
            node_color = '#9999FF'  
        
        # 使用Ellipse绘制，分别设置width和height，使其在显示时为正圆
        ellipse = Ellipse((x, y), width_data, height_data,
                         color=node_color,
                         alpha=0.8,
                         edgecolor='black',
                         linewidth=2,
                         zorder=3)
        ax.add_patch(ellipse)
        
        # 添加城市标签（使用简称）
        city_row = df[df['全称'] == city_name]
        if not city_row.empty and pd.notna(city_row.iloc[0].get('简称')):
            label = city_row.iloc[0]['简称']
        else:
            label = city_name
        
        ax.text(x, y, label, 
               fontsize=20,
               fontweight='bold',
               ha='center',
               va='center',
               color='white' if abs_value > (min_abs + max_abs) / 2 else 'black',
               zorder=4)

# 注意：坐标轴范围和位置已在绘制节点前设置，这里不再重复设置

# 添加图例
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D

# 创建图例元素
legend_elements = [
    Circle((0, 0), 0.1, facecolor='#FF9900',alpha=0.8, edgecolor='black', label=f'{legend_label}>0'),
    Circle((0, 0), 0.1, facecolor='#9999FF',alpha=0.8, edgecolor='black', label=f'{legend_label}<0'),
    Line2D([0], [0], color='gray', linewidth=MIN_LINE_WIDTH, label=f'Weak link of PL (Smaller: {min_allocation:.3f})'),
    Line2D([0], [0], color='gray', linewidth=MAX_LINE_WIDTH, label=f'Strong link of PL (Larger: {max_allocation:.3f})'),
    Patch(facecolor='none', edgecolor='none', label='Node Size: Dpe Absolute Value')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=15, frameon=False)

# 设置标题（使用transAxes确保标题在图幅范围内）
ax.set_title(f'{DATA_COLUMN} Population-Land Network Analysis',
             fontsize=20, fontweight='bold', pad=20)

# 移除坐标轴
ax.set_axis_off()

# 保存图片（与gisDpeanalysis.py保持一致）
print(f"   - 正在保存图片: {OUTPUT_PNG}")
plt.savefig(OUTPUT_PNG, 
            dpi=300, 
            bbox_inches=None,  # 使用None，配合set_position确保图像尺寸一致
            pad_inches=0,  # 无额外边距
            facecolor='white',
            edgecolor='none',
            format='png')
print(f"   - 图片已保存: {OUTPUT_PNG}")

# 输出统计信息
print("\n" + "="*80)
print("网络分析统计:")
print("="*80)
if HAS_NETWORKX:
    print(f"总节点数: {G.number_of_nodes()}")
    print(f"总边数: {G.number_of_edges()}")
else:
    print(f"总节点数: {len(all_cities)}")
    print(f"总边数: {len(edges)}")
print(f"\n入度统计:")
for city, degree in sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {city}: {degree}")
print(f"\n出度统计:")
for city, degree in sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {city}: {degree}")

plt.close()

print("\n处理完成！")

