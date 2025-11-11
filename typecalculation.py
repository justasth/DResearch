import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import plotly.graph_objects as go

EXCEL_PATH = "analysisdata.xlsx"
data = pd.read_excel(EXCEL_PATH)

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
        # 当Dpl<0, Dpe>0时，Dle = Dpl - Dpe < 0，但数据中可能有Dle>0的情况（数据不一致）
        # 两种情况都归入象限2
        return "象限2"
    # 象限3-1: Dpl<0, Dpe<0, Dle>0
    elif dpl_neg and dpe_neg and dle_pos:
        return "象限3-1"
    # 象限3-2: Dpl<0, Dpe<0, Dle<0
    elif dpl_neg and dpe_neg and dle_neg:
        return "象限3-2"
    # 象限4: Dpl>0, Dpe<0, Dle<0 或 Dpl>0, Dpe<0, Dle>0
    elif dpl_pos and dpe_neg:
        # 当Dpl>0, Dpe<0时，Dle = Dpl - Dpe > 0，但数据中可能有Dle<0的情况（数据不一致）
        # 两种情况都归入象限4
        return "象限4"
    else:
        # 打印调试信息
        print(f"警告: 未分类的情况 - Dpl={dpl}, Dpe={dpe}, Dle={dle}, 计算的Dle={dpl-dpe}")
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

# 象限标签映射
quadrant_labels = {
    "象限1-1": "产不足+地紧缺",
    "象限1-2": "地冗余+产匮乏",
    "象限2": "地过溢+产匮乏",
    "象限3-1": "地过溢+产较强",
    "象限3-2": "产过强+地冗余",
    "象限4": "产过强+地紧缺"
}

# 为每个象限定义不同的颜色
quadrant_colors = {
    "象限1-1": "#9888CB",
    "象限1-2": "#CCC3E6",
    "象限2": "#E5CDCC",
    "象限3-1": "#F59300",
    "象限3-2": "#B75B00",
    "象限4": "#995B01"
}

# 处理所有年份的数据
years = [1990, 2000, 2010, 2020]
quadrant_data = {}

for year in years:
    dpl_col = f"Dpl{year}"
    dpe_col = f"Dpe{year}"
    
    if dpl_col not in data.columns or dpe_col not in data.columns:
        print(f"警告: 未找到 {dpl_col} 或 {dpe_col} 列")
        continue
    
    dpl = data[dpl_col]
    dpe = data[dpe_col]
    dle = dpl - dpe  # 根据注释：Dle = Dpl - Dpe
    
    # 对每个城市进行分类
    quadrants = []
    unclassified_cities = []
    for i in range(len(data)):
        q = classify_quadrant(dpl.iloc[i], dpe.iloc[i], dle.iloc[i])
        quadrants.append(q)
        if q == "未分类":
            city_name = data.iloc[i].get('全称', f'城市{i+1}')
            unclassified_cities.append({
                '城市': city_name,
                'Dpl': dpl.iloc[i],
                'Dpe': dpe.iloc[i],
                'Dle': dle.iloc[i]
            })
    
    data[f'象限{year}'] = quadrants
    
    # 统计每个象限的城市数量
    quadrant_counts = pd.Series(quadrants).value_counts().sort_index()
    quadrant_data[year] = quadrant_counts
    total_classified = quadrant_counts.sum()
    print(f"\n{year}年象限统计:")
    for q, count in quadrant_counts.items():
        label = quadrant_labels.get(q, q)
        print(f"  {q} ({label}): {count}个城市")
    
    # 验证总数
    print(f"  总计: {total_classified}个城市")
    if total_classified != 41:
        print(f"  警告: 城市总数不是41个，实际为{total_classified}个")
    
    # 如果有未分类的城市，打印详细信息
    if unclassified_cities:
        print(f"\n  警告: {year}年有{len(unclassified_cities)}个城市未分类:")
        for city_info in unclassified_cities:
            print(f"    {city_info['城市']}: Dpl={city_info['Dpl']:.4f}, Dpe={city_info['Dpe']:.4f}, Dle={city_info['Dle']:.4f}")

# 创建桑基图数据
# 准备节点和连接
all_quadrants = ["象限1-1", "象限1-2", "象限2", "象限3-1", "象限3-2", "象限4"]

# 创建节点标签（包含年份）
nodes_1990 = [f"1990-{q}" for q in all_quadrants]
nodes_2000 = [f"2000-{q}" for q in all_quadrants]
nodes_2010 = [f"2010-{q}" for q in all_quadrants]
nodes_2020 = [f"2020-{q}" for q in all_quadrants]

all_nodes = nodes_1990 + nodes_2000 + nodes_2010 + nodes_2020

# 创建连接数据（source, target, value, 城市列表）
# 为每个城市创建独立的流，确保41个城市对应41条流
links = []
link_city_names = []  # 存储每个连接对应的城市简称

# 检查是否有简称列
has_abbrev = '简称' in data.columns

# 1990→2000的连接：为每个城市创建独立的流
for city_idx in data.index:
    q1 = data.loc[city_idx, f'象限1990']
    q2 = data.loc[city_idx, f'象限2000']
    
    # 找到源节点和目标节点的索引
    if q1 in all_quadrants and q2 in all_quadrants:
        source_idx = all_quadrants.index(q1)
        target_idx = len(nodes_1990) + all_quadrants.index(q2)
        
        # 获取城市简称
        if has_abbrev:
            city_name = str(data.loc[city_idx, '简称']) if pd.notna(data.loc[city_idx, '简称']) else f"城市{city_idx+1}"
        else:
            city_name = f"城市{city_idx+1}"
        
        # 为每个城市创建一条流（value=1）
        links.append({
            'source': source_idx,
            'target': target_idx,
            'value': 1
        })
        link_city_names.append(city_name)

# 2000→2010的连接：为每个城市创建独立的流
for city_idx in data.index:
    q1 = data.loc[city_idx, f'象限2000']
    q2 = data.loc[city_idx, f'象限2010']
    
    # 找到源节点和目标节点的索引
    if q1 in all_quadrants and q2 in all_quadrants:
        source_idx = len(nodes_1990) + all_quadrants.index(q1)
        target_idx = len(nodes_1990) + len(nodes_2000) + all_quadrants.index(q2)
        
        # 获取城市简称
        if has_abbrev:
            city_name = str(data.loc[city_idx, '简称']) if pd.notna(data.loc[city_idx, '简称']) else f"城市{city_idx+1}"
        else:
            city_name = f"城市{city_idx+1}"
        
        # 为每个城市创建一条流（value=1）
        links.append({
            'source': source_idx,
            'target': target_idx,
            'value': 1
        })
        link_city_names.append(city_name)

# 2010→2020的连接：为每个城市创建独立的流
for city_idx in data.index:
    q1 = data.loc[city_idx, f'象限2010']
    q2 = data.loc[city_idx, f'象限2020']
    
    # 找到源节点和目标节点的索引
    if q1 in all_quadrants and q2 in all_quadrants:
        source_idx = len(nodes_1990) + len(nodes_2000) + all_quadrants.index(q1)
        target_idx = len(nodes_1990) + len(nodes_2000) + len(nodes_2010) + all_quadrants.index(q2)
        
        # 获取城市简称
        if has_abbrev:
            city_name = str(data.loc[city_idx, '简称']) if pd.notna(data.loc[city_idx, '简称']) else f"城市{city_idx+1}"
        else:
            city_name = f"城市{city_idx+1}"
        
        # 为每个城市创建一条流（value=1）
        links.append({
            'source': source_idx,
            'target': target_idx,
            'value': 1
        })
        link_city_names.append(city_name)

# 创建节点标签（显示象限名称和标签）
node_labels = []
node_colors = []
node_x = []  # x坐标（按年份）
node_y = []  # y坐标（按象限顺序，紧密排列）

# 计算每个节点的位置和颜色
# 首先计算每个年份的总城市数，用于归一化
for year_idx, year in enumerate(years):
    # 计算该年份每个象限的城市数量
    year_quadrant_counts = {}
    total_count = 0
    for q in all_quadrants:
        count = (data[f'象限{year}'] == q).sum()
        year_quadrant_counts[q] = count
        total_count += count
    
    # 按照象限顺序（1-1, 1-2, 2, 3-1, 3-2, 4）固定排列节点
    # 每个象限占据固定的位置，从上到下排列
    for q_idx, q in enumerate(all_quadrants):
        count = year_quadrant_counts.get(q, 0)
        
        # x坐标：按年份分布，减少间距（使用更紧密的分布）
        # 将间距从均匀分布改为更紧密的分布（0.1, 0.35, 0.6, 0.85）
        if len(years) > 1:
            # 使用更紧密的间距
            x_positions = [0.1, 0.2, 0.3, 0.4]  # 对应1990, 2000, 2010, 2020
            if year_idx < len(x_positions):
                node_x.append(x_positions[year_idx])
            else:
                node_x.append(year_idx / (len(years) - 1))
        else:
            node_x.append(0.5)
        
        # y坐标：按照象限顺序固定排列，每个象限占据1/6的位置
        # 从上到下：象限1-1在最上面，象限4在最下面
        # 使用反向索引，使象限1-1在顶部（y值大），象限4在底部（y值小）
        node_y.append(1.0 - (q_idx + 0.5) / len(all_quadrants))
        
        # 标签和颜色
        count = year_quadrant_counts.get(q, 0)
        label = quadrant_labels.get(q, q)  # 获取象限的中文标签,象限1-1为产不足+地紧缺,象限1-2为地冗余+产匮乏,象限2为地过溢+产匮乏,象限3-1为地过溢+产较强,象限3-2为产过强+地冗余,象限4为产过强+地紧缺
        
        # 提取象限编号（从"象限1-1"中提取"1-1"）
        quadrant_num = q.replace("Quadrant", "")  # 移除"象限"前缀，得到"1-1"、"1-2"等
        
        # 节点标签：显示象限编号、象限中文标签和数量
        node_labels.append(f"{quadrant_num}\n{label}\n{count}")
        
        node_colors.append(quadrant_colors.get(q, "lightgray"))

# 为连接线设置颜色（使用源节点的颜色，但透明度更高）
link_colors = []
for link in links:
    source_idx = link['source']
    source_color = node_colors[source_idx]
    # 将颜色转换为rgba格式，设置透明度
    if source_color.startswith('#'):
        # 将十六进制颜色转换为rgba
        r = int(source_color[1:3], 16)
        g = int(source_color[3:5], 16)
        b = int(source_color[5:7], 16)
        link_colors.append(f"rgba({r},{g},{b},0.3)")
    else:
        link_colors.append("rgba(0,0,255,0.2)")

# 使用plotly创建桑基图
# 注意：plotly的Sankey图会自动排列节点，我们通过设置pad=0来减少间隔
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=0,  # 设置pad为0，使节点之间没有间隔
        thickness=40,  # 增加节点厚度以容纳更多文字
        line=dict(color="black", width=1),
        label=node_labels,
        color=node_colors
    ),
    link=dict(
        source=[link['source'] for link in links],
        target=[link['target'] for link in links],
        value=[link['value'] for link in links],
        color=link_colors,
        label=link_city_names,  # 在连接线上显示城市简称
        hovertemplate='%{label}<br>城市数量: %{value}<extra></extra>'
    )
)])

# 不添加任何标签
annotations = []

fig.update_layout(
    title_text="1990-2000-2010-2020 Year Qudrant Change Sankey Diagram",
    font_size=20,
    width=1600,
    height=900,
    annotations=annotations,
    margin=dict(b=40)  # 减少底部边距
)

# 保存为HTML文件
fig.write_html("象限变化桑基图.html")
print("\n桑基图已保存为: 象限变化桑基图.html")

# 保存为PNG图片（需要安装kaleido: pip install kaleido）
try:
    # 提高分辨率以获得更清晰的图片
    fig.write_image("象限变化桑基图.png", width=2400, height=1350, scale=2)
    print("桑基图已保存为: 象限变化桑基图.png (2400x1350, 2x scale)")
except ImportError:
    print("\n警告: 未安装kaleido，无法保存PNG图片")
    print("请运行: pip install kaleido")
    print("然后重新运行此脚本以保存PNG图片")
except Exception as e:
    print(f"\n保存PNG失败: {e}")
    print("如果未安装kaleido，请运行: pip install kaleido")

# 显示图表
fig.show()
