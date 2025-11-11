import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.font_manager as fm
font = fm.FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc')

# 配置参数
filename = "provinceanalysisdata.xlsx"  # Excel文件名
x_col = "Dpl2020"               # x轴列名
y_col = "Dpe2020"               # y轴列名
size_col = "p2020"              # 点大小列名
label_col = "city"              # 标签列名
def_title = f"Dpl_2020 vs Dpe_2020" + "\nPoint Size: Population_2020"
def_filename = "provinceDplDelscatter2020.png"

# 全局点大小范围 - 确保不同图表间可比性
MIN_POINT_SIZE = 10
MAX_POINT_SIZE = 1000

# 固定的人口范围 - 基于所有时间段的数据确定
GLOBAL_MIN_POP = 0              # 最小人口
GLOBAL_MAX_POP = 2500       # 最大人口

def map_sizes(sizes, min_pop=GLOBAL_MIN_POP, max_pop=GLOBAL_MAX_POP, 
             min_size=MIN_POINT_SIZE, max_size=MAX_POINT_SIZE):
    """
    使用线性映射将人口数据映射到点大小范围
    
    参数
    ----------
    sizes : array-like
        人口数据
    min_pop : float
        最小人口
    max_pop : float
        最大人口
    min_size : float
        最小点大小
    max_size : float
        最大点大小
    
    返回
    -------
    array-like
        映射后的点大小
    """
    # 线性映射到指定大小范围
    sizes = np.array(sizes)
    mapped = min_size + (sizes - min_pop) * (max_size - min_size) / (max_pop - min_pop)
    
    # 限制在范围内
    mapped = np.clip(mapped, min_size, max_size)
    return mapped

####################################################################################
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', edgecolor='red', linestyle='--', **kwargs):
    """
    创建置信椭圆图

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radius.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    # 计算协方差矩阵
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    # 使用卡方分布获得95%置信区间的半径（2维）
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor=edgecolor, linestyle=linestyle, **kwargs)
    
    # 计算椭圆中心（均值）
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # 计算椭圆的方向（角度）
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    a=ell_radius_x
    b=ell_radius_y
    
    return ax.add_patch(ellipse), a, b

####################################################################################
def main():
    # 读取并处理数据
    df = pd.read_excel(filename)
    # 转换数值类型并处理无效值
    x = df[x_col]
    y = df[y_col]
    sizes = df[size_col]
    labels = df[label_col]
    # 根据人口数量设置不同的alpha值
    alpha_values = np.where(sizes > 500, 0.2, 0)
    # 创建图表
    plt.figure(figsize=(10, 8), dpi=150)
    ax = plt.gca()

    # 更可靠的颜色处理方式
    base_color = mcolors.to_rgb("black")  # 转换为RGB元组
    fill_colors = [(*base_color, alpha) for alpha in alpha_values]

    scaled_sizes = map_sizes(sizes)

    # 绘制散点图
    scatter = ax.scatter(
        x, 
        y, 
        s=scaled_sizes,
        facecolors=fill_colors, # 填充使用带透明度的颜色
        edgecolors="black", # 描边使用完全不透明的白色
        linewidth=0.8
    )
    # 添加置信椭圆（95%置信区间，n_std=2）
    ellipse_patch, a, b=confidence_ellipse(x, y, ax, n_std=2, edgecolor='black', linestyle='--', linewidth=0.8, alpha=0.8)

    # 添加城市标签
    for i, label in enumerate(labels):
        # 排除"上海"标签，因为它在图纸范围以外
        # if label != 1:
            ax.text(x[i], y[i], label, fontsize=10, ha='center', va='center', color='black',fontproperties=font)

    # 绘制y=x直线
    ax.plot([-30, 30], [-30, 30], color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # 添加标签和标题
    ax.set_xlabel(f"Dpl %", fontsize=15, labelpad=10)
    ax.set_ylabel(f"Dpe %", fontsize=15, labelpad=10)

    ax.set_title(def_title, fontsize=15, pad=15)
    
    # 设置坐标轴范围
    ax.set_aspect('equal')
    ax.set_xlim(-26, 26)#设置坐标轴字号和间距
    ax.set_ylim(-26, 26)
    ax.set_xticks(np.arange(-25, 26, 5))
    ax.set_yticks(np.arange(-25, 26, 5))
    ax.set_xticklabels(np.arange(-25, 26, 5), fontsize=15)
    ax.set_yticklabels(np.arange(-25, 26, 5), fontsize=15)
    
    Dpl = np.mean(np.abs(x))
    Dpe = np.mean(np.abs(y))
    
    ax.text(-25, -25, f"a={a:.2f}, b={b:.2f} \n$\overline{{Dpl}}={Dpl:.2f},\overline{{Dpe}}={Dpe:.2f}$", fontsize=15, ha='left', va='bottom')
    
    # 添加参考线
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=1)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=1)
    
    # 网格和样式优化
    ax.grid(True, linestyle=':', color='gray', alpha=0.4)
    ax.set_axisbelow(True)  # 网格线在数据下方

    # 保存和显示
    plt.tight_layout()
    plt.savefig(def_filename, bbox_inches='tight', dpi=300)
    plt.show()

####################################################################################
if __name__ == "__main__":
    main()
    
    