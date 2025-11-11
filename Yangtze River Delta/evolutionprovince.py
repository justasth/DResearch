import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 配置参数
filename = "provinceanalysisdata.xlsx"  # Excel文件名
df = pd.read_excel(filename)

city_name = [
    'Shanghai', 'Jiangsu', 'Zhejiang', 'Anhui'
]

def plot_line_chart(def_title, def_filename, a):
    """绘制折线图"""
    try:
        # 读取Excel文件，指定第一行为列名
        df = pd.read_excel(filename, header=0)  # header=0表示第一行作为列名
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 打印所有列名以便调试
    print("所有列名:", df.columns.tolist())
    
    # 定义需要提取的列（N-Q列）
    # 在Excel中，列名通常对应字母，但pandas会使用原始列名
    # 这里假设N-Q列对应的列名是 'N', 'O', 'P', 'Q'
    # 或者可能是 '1990', '2000', '2010', '2020' 等年份格式
    # 根据实际列名修改下面的列表
    year_columnsDpl = ['Dpl1990', 'Dpl2000', 'Dpl2010', 'Dpl2020']  # 或者 ['1990', '2000', '2010', '2020']
    year_columnsDpe = ['Dpe1990', 'Dpe2000', 'Dpe2010', 'Dpe2020']  # 或者 ['1990', '2000', '2010', '2020']
    
    # 验证列名是否存在
    valid_columnsDpl = [col for col in year_columnsDpl if col in df.columns]
    valid_columnsDpe = [col for col in year_columnsDpe if col in df.columns]
    
    if not valid_columnsDpl or not valid_columnsDpe:
        print(f"错误: 未找到列名 {year_columnsDpl} 或 {year_columnsDpe}")
        print("可用的列名:", df.columns.tolist())
        return
    
    # 提取第二行数据（索引为1，因为索引0是第一行列名）
    if len(df) < 2:
        print("错误: 文件至少需要2行数据（列名行+数据行）")
        return
    
    # 提取第二行的数据
    row_index = a  # 第二行（索引1）
    year_dataDpl = df.loc[row_index, year_columnsDpl].values
    year_dataDpe = df.loc[row_index, year_columnsDpe].values
    
    # 自定义横坐标
    years = [1990, 2000, 2010, 2020]
    
    # 创建图表
    plt.figure(figsize=(10, 8), dpi=150)
    
    # 绘制Dpl折线图
    plt.plot(years, year_dataDpl, 
            #  marker='o', 
            #  markersize=8,
            linestyle='-',
             linewidth=2.5,
             color='black',
            #  markerfacecolor='white',
             markeredgewidth=2,
            #  markeredgecolor='black',
             label='Dpl')
    # 绘制面积图
    plt.fill_between(years, year_dataDpl, color='#FF9900', alpha=0.3)
    # 绘制面积图
    plt.fill_between(years, year_dataDpe, color='#9999FF', alpha=0.3)
    # # 绘制柱状图
    # plt.bar(years, year_dataDpl, 
    #         width=0.5,
    #         color='black',
    #         alpha=0.5)
    
    # 添加Dpl数据点标签
    for i, (x, y) in enumerate(zip(years, year_dataDpl)):
        plt.text(x, y, f"{y:.2f}", 
                 fontsize=15, 
                 ha='center', 
                 va='bottom',
                 bbox=dict(facecolor='white', alpha=0, edgecolor='lightgray', pad=4))
    
    # 绘制Dpe折线图
    plt.plot(years, year_dataDpe, 
            #  marker='o', 
            #  markersize=8,
             linewidth=2.5,
             color='black',
             linestyle='--',
            #  markerfacecolor='white',
             markeredgewidth=2,
            #  markeredgecolor='black',
             label='Dpe')
    
    # 添加Dpe数据点标签
    for i, (x, y) in enumerate(zip(years, year_dataDpe)):
        plt.text(x, y, f"{y:.2f}", 
                 fontsize=15, 
                 ha='center', 
                 va='top',
                 bbox=dict(facecolor='white', alpha=0, edgecolor='lightgray', pad=4))

    # 添加标题和标签
    plt.title(def_title, fontsize=25, pad=15)
    plt.xlabel("Year", fontsize=20, labelpad=10)
    plt.ylabel("Dpe/Dpl %", fontsize=20, labelpad=10)
    
    # 设置x轴刻度
    plt.xticks(years, [str(year) for year in years], fontsize=20)
    plt.ylim(-5, 5)
    plt.yticks(np.arange(-15, 25, 5), fontsize=20)#############
    
    # 设置网格和样式
    plt.grid(True, linestyle='--', alpha=0.7, color='lightgray')
    plt.axhline(y=0, color='black', linewidth=0.8, alpha=1)  # 添加0参考线
    
    # 添加图例
    plt.legend(fontsize=20, loc='upper right')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(def_filename, dpi=300, bbox_inches='tight')
    print(f"折线图已保存为: {def_filename}")
    # plt.show()

if __name__ == "__main__":
    for city in df['city']:
        city_title = city_name[city-1]
        def_title = f"{city_title}"  # 折线图标题
        def_filename = f"{city}{city_title}_evolution.png"  # 折线图文件名
        a =city-1  # 第二行（索引1）
        
        # def_title = "Shanghai"  # 折线图标题
        # def_filename = "1Shanghai_evolution.png"  # 折线图文件名
        # a =0  # 第二行（索引1）
        plot_line_chart(def_title, def_filename, a)