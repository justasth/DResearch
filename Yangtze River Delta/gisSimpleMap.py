import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from shapely.geometry import box

# 尝试导入rasterio用于读取高程数据
try:
    import rasterio
    from rasterio.plot import show
    from rasterio.warp import transform as rasterio_transform, calculate_default_transform, reproject, Resampling
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("警告: rasterio未安装，将跳过高程显示。可以使用 'pip install rasterio' 安装。")

# 配置参数
SHP_PATH = "data/长三角市级边界.shp"
RIVER_PATH = "data/riveryzr.shp"
DEM_PATH = "data/长三角高程.tif"
EXCEL_PATH = "analysisdata.xlsx"
OUTPUT_PNG = "长三角地级市边界图.png"

# 统一的坐标轴范围（与其他地图保持一致）
UNIFIED_X_MIN = 114.878463
UNIFIED_Y_MIN = 27.143423
UNIFIED_X_MAX = 122.834203
UNIFIED_Y_MAX = 35.127197

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

# 1. 加载行政边界数据
print("\n[步骤1/6] 加载行政边界数据...")
gdf = gpd.read_file(SHP_PATH, engine='fiona', encoding='utf-8')

# 检查坐标参考系统(CRS)，如果不是WGS84(EPSG:4326)则转换
if gdf.crs is None or (hasattr(gdf.crs, 'to_epsg') and gdf.crs.to_epsg() != 4326):
    print("   - 转换坐标系到WGS84 (EPSG:4326)...")
    gdf = gdf.to_crs(epsg=4326)

print(f"   - 边界数据加载完成，包含 {len(gdf)} 个要素")

# 2. 加载Excel数据
print("\n[步骤2/6] 加载Excel数据...")
df = pd.read_excel(EXCEL_PATH)
print(f"   - Excel数据加载完成，包含 {len(df)} 行")

# 3. 合并地理数据与Excel数据（获取简称）
print("\n[步骤3/6] 合并地理数据和Excel数据...")
merged_gdf = gdf.merge(df, left_on='地名', right_on='全称', how='left')

# 添加省份信息
merged_gdf['province'] = merged_gdf['地名'].apply(get_province)

# 检查合并结果
matched_count = merged_gdf['简称'].notna().sum()
print(f"   - 成功匹配简称: {matched_count}/{len(merged_gdf)}")

# 4. 加载水系数据
print("\n[步骤4/6] 加载水系数据...")
try:
    river_gdf = gpd.read_file(RIVER_PATH, engine='fiona', encoding='utf-8')
    if river_gdf.crs is None or (hasattr(river_gdf.crs, 'to_epsg') and river_gdf.crs.to_epsg() != 4326):
        print("   - 转换水系坐标系到WGS84 (EPSG:4326)...")
        river_gdf = river_gdf.to_crs(epsg=4326)
    # 裁剪到研究区域
    bbox = box(UNIFIED_X_MIN, UNIFIED_Y_MIN, UNIFIED_X_MAX, UNIFIED_Y_MAX)
    river_gdf = river_gdf.clip(bbox)
    print(f"   - 水系数据加载完成，包含 {len(river_gdf)} 个要素")
    RIVER_AVAILABLE = True
except Exception as e:
    print(f"   - 警告: 无法加载水系数据: {e}")
    RIVER_AVAILABLE = False
    river_gdf = None

# 5. 加载高程数据
print("\n[步骤5/6] 加载高程数据...")
dem_data = None
dem_extent = None
DEM_AVAILABLE = False

if RASTERIO_AVAILABLE:
    try:
        print(f"   - 尝试打开高程文件: {DEM_PATH}")
        with rasterio.open(DEM_PATH) as src:
            print(f"   - 文件打开成功，CRS: {src.crs}, 尺寸: {src.width}x{src.height}")
            print(f"   - 原始边界: {src.bounds}")
            
            # 直接读取整个数据集（已经是WGS84）
            dem_data = src.read(1)
            
            # 保存原始的transform用于后续mask操作
            dem_transform = src.transform
            
            # 使用原始边界作为extent [left, right, bottom, top]
            dem_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            
            print(f"   - 高程数据加载完成，尺寸: {dem_data.shape}")
            print(f"   - 数据范围: {dem_extent}")
            print(f"   - 高程值范围: {np.nanmin(dem_data):.2f} - {np.nanmax(dem_data):.2f}")
            DEM_AVAILABLE = True
                
    except Exception as e:
        print(f"   - 警告: 无法加载高程数据: {e}")
        import traceback
        traceback.print_exc()
        DEM_AVAILABLE = False
else:
    print("   - rasterio未安装，无法加载高程数据")
    print("   - 提示: 可以运行 'pip install rasterio' 安装")

# 6. 创建地图
print("\n[步骤6/6] 创建地图...")

# 创建图形
fig, ax = plt.subplots(figsize=(16, 12), dpi=300)

# 设置透明背景
fig.patch.set_facecolor('none')

# 绘制高程（底层）
if DEM_AVAILABLE and dem_data is not None and dem_extent is not None:
    print("   - 正在绘制高程数据...")
    
    # 创建高程数据的副本用于处理（转换为浮点数以支持NaN）
    dem_data_plot = dem_data.copy().astype(np.float64)
    
    # 先不处理负值，保留原始数据，等mask后再处理
    # 只处理NaN值
    if np.isnan(dem_data).any():
        dem_data_plot[np.isnan(dem_data)] = np.nan
    
    # 创建边界mask：边界外的区域设为NaN（透明）
    # 使用长三角地区边界来创建mask
    if 'geometry' in merged_gdf.columns:
        try:
            # 确保region_union的坐标系与高程数据一致（WGS84）
            region_gdf = gpd.GeoDataFrame([1], geometry=[merged_gdf.unary_union], crs=merged_gdf.crs)
            if region_gdf.crs is None or (hasattr(region_gdf.crs, 'to_epsg') and region_gdf.crs.to_epsg() != 4326):
                region_gdf = region_gdf.to_crs(epsg=4326)
            region_union = region_gdf.geometry.iloc[0]
            
            # 必须使用原始的rasterio transform，否则mask位置会错
            # 重新打开文件获取transform
            with rasterio.open(DEM_PATH) as src:
                transform = src.transform
                print(f"   - 使用原始transform: {transform}")
            
            # 创建mask
            # geometry_mask默认行为（invert=False）：边界内为True，边界外为False
            # 我们需要：边界内保留数据，边界外设为NaN
            # 所以：边界外为True（设为NaN），边界内为False（保留数据）
            mask_inside = geometry_mask(
                [region_union],
                out_shape=dem_data.shape,
                transform=transform,
                invert=False  # invert=False: 边界内为True，边界外为False
            )
            
            # 反转得到边界外mask：边界外为True（设为NaN），边界内为False（保留）
            mask_outside = ~mask_inside
            
            # 检查mask是否正确
            mask_inside_ratio = np.sum(mask_inside) / mask_inside.size
            print(f"   - Mask检查: 边界内像素比例: {mask_inside_ratio:.2%}")
            
            # 检查边界内数据的统计信息（在mask之前，使用原始dem_data）
            inside_data_before = dem_data[mask_inside]
            print(f"   - 边界内原始数据统计: 最小值={np.nanmin(inside_data_before):.1f}, 最大值={np.nanmax(inside_data_before):.1f}, 正值数量={np.sum(inside_data_before > 0)}")
            
            # 如果边界内数据都是0，说明mask位置错了，尝试反转
            if np.nanmax(inside_data_before) == 0 and np.sum(inside_data_before > 0) == 0:
                print(f"   - 警告: 边界内数据都是0，尝试反转mask")
                mask_outside = mask_inside  # 反转：原来边界内变成边界外
                mask_inside = ~mask_outside
                inside_data_before = dem_data[mask_inside]
                print(f"   - 反转后边界内数据统计: 最小值={np.nanmin(inside_data_before):.1f}, 最大值={np.nanmax(inside_data_before):.1f}, 正值数量={np.sum(inside_data_before > 0)}")
            
            # 将边界外的区域设为NaN（透明）
            dem_data_plot[mask_outside] = np.nan
            
            # 现在处理边界内的负值：将负值设为0（海平面或无效数据）
            dem_data_plot[(dem_data_plot < 0) & (~np.isnan(dem_data_plot))] = 0
            
            valid_count = np.sum(~np.isnan(dem_data_plot))
            positive_count = np.sum((dem_data_plot > 0) & (~np.isnan(dem_data_plot)))
            print(f"   - 已将边界外区域设为NaN，边界内有效像素: {valid_count}, 正值像素: {positive_count}")
        except Exception as e:
            print(f"   - 警告: 创建边界mask失败，将显示全部数据: {e}")
    
    # 计算实际的高程值范围（排除NaN和0值，因为0可能是边界外的值）
    valid_elevation = dem_data_plot[~np.isnan(dem_data_plot)]
    # 进一步筛选：排除0值（0可能是边界外或海平面）
    valid_elevation_positive = valid_elevation[valid_elevation > 0]
    
    if len(valid_elevation_positive) > 0:
        elev_min = np.nanmin(valid_elevation_positive)
        elev_max = np.nanmax(valid_elevation_positive)
    elif len(valid_elevation) > 0:
        # 如果没有正值，使用所有非NaN值
        elev_min = np.nanmin(valid_elevation)
        elev_max = np.nanmax(valid_elevation)
    else:
        # 如果全部是NaN，使用原始数据的范围
        elev_min = 0
        elev_max = np.nanmax(dem_data[~np.isnan(dem_data)])
    
    print(f"   - 高程值范围: {elev_min:.1f} - {elev_max:.1f} m")
    print(f"   - 有效高程值统计: 总数={len(valid_elevation)}, 正值={len(valid_elevation_positive) if len(valid_elevation) > 0 else 0}")
    
    # 创建高程颜色映射（从高到低：深色到浅色/白色）
    # 高海拔用深色，低海拔用浅色或白色
    colors = ['#FFFFFF', '#F5F5DC', '#E6E6B8', '#D4D4A0', '#C2C288', '#B0B070', '#9E9E58', '#8C8C40', '#7A7A28', '#5A5A18']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('elevation', colors, N=n_bins)
    
    # 创建masked array用于显示（NaN值会透明）
    dem_data_masked = np.ma.masked_invalid(dem_data_plot)
    
    # 显示高程，extent格式为 [left, right, bottom, top]
    # 从高值到低值映射：高海拔深色，低海拔浅色
    im = ax.imshow(
        dem_data_masked,
        extent=dem_extent,  # [left, right, bottom, top]
        cmap=cmap,
        alpha=0.7,  # 稍微提高透明度，让山更明显
        interpolation='bilinear',
        zorder=0,
        origin='upper',  # 确保图像方向正确
        vmin=elev_min,  # 最低值（白色/透明）
        vmax=elev_max   # 最高值（深色）
    )
    
    print(f"   - 高程数据已绘制，范围: {dem_extent}")
    print(f"   - 高程数据形状: {dem_data.shape}, 有效值数量: {len(valid_elevation)}")
else:
    if not DEM_AVAILABLE:
        print("   - 跳过高程显示（rasterio未安装或数据加载失败）")
    elif dem_data is None:
        print("   - 跳过高程显示（数据为None）")
    elif dem_extent is None:
        print("   - 跳过高程显示（extent为None）")

# 绘制水系（第二层）
if RIVER_AVAILABLE and river_gdf is not None:
    river_gdf.plot(
        ax=ax,
        color='#4A90E2',  # 蓝色
        linewidth=0.8,
        alpha=0.5,
        zorder=2
    )

# 绘制地级市边界（黑色细线）
merged_gdf.plot(
    ax=ax,
    edgecolor='black',  # 黑色
    facecolor='none',
    linewidth=0.5,  # 细线
    zorder=3
)

# 绘制省级边界（浅紫色）
if 'province' in merged_gdf.columns:
    province_gdf = merged_gdf[merged_gdf['province'].notna()].copy()
    if not province_gdf.empty:
        # 合并同一省份的城市
        
        province_bounds = province_gdf.dissolve(by='province')

        province_bounds.plot(
            ax=ax,
            facecolor='none',
            edgecolor='#DDA0DD',  # 浅紫色
            linewidth=2.5,  # 加粗
            linestyle='-',
            zorder=4
        )
        
        province_bounds.plot(
            ax=ax,
            facecolor='none',
            edgecolor='black',  # 浅紫色
            linewidth=1,  # 加粗
            linestyle='--',
            zorder=4
        )

# 绘制长三角地区边界（最外层，深紫色+浅紫色双层）
# 创建整个区域的边界
region_bbox = box(UNIFIED_X_MIN, UNIFIED_Y_MIN, UNIFIED_X_MAX, UNIFIED_Y_MAX)
region_gdf = gpd.GeoDataFrame([1], geometry=[region_bbox], crs=gdf.crs)

# # 先绘制浅紫色更粗边界（外层）
# region_gdf.plot(
#     ax=ax,
#     edgecolor='#DDA0DD',  # 浅紫色
#     facecolor='none',
#     linewidth=5.0,  # 更粗
#     alpha=0.2,
#     zorder=5
# )
# # 再绘制深紫色加粗边界（内层）
# region_gdf.plot(
#     ax=ax,
#     edgecolor='#9370DB',  # 深紫色
#     facecolor='none',
#     linewidth=3.0,  # 加粗
#     zorder=6
# )

# 添加城市简称标签（最上层）
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
                    zorder=500
                )
            except:
                pass

# 设置坐标轴范围（与其他地图保持一致）
ax.set_xlim(UNIFIED_X_MIN, UNIFIED_X_MAX)
ax.set_ylim(UNIFIED_Y_MIN, UNIFIED_Y_MAX)

# 设置坐标轴位置（居中，为colorbar留出空间）
# 如果有高程数据，为colorbar留出右侧空间；否则居中
if DEM_AVAILABLE and dem_data is not None and dem_extent is not None:
    # 为colorbar留出右侧空间，地图稍微左移以居中
    ax.set_position([0.05, 0.05, 0.80, 0.90])
    
    # 添加高程colorbar（图例），显示从最低到最高值的映射
    # 调整colorbar位置，避免叠加在地图上
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.6, aspect=30)
    cbar.set_label('Elevation (m)', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)
else:
    # 没有高程数据时居中
    ax.set_position([0.05, 0.05, 0.90, 0.90])

# 移除坐标轴刻度
ax.set_xticks([])
ax.set_yticks([])

# 移除坐标轴边框
for spine in ax.spines.values():
    spine.set_visible(False)

# 移除所有边框
ax.set_frame_on(False)

# 添加图例（不包括高程，因为高程使用colorbar显示）
legend_elements = []

# 水系图例
if RIVER_AVAILABLE and river_gdf is not None:
    legend_elements.append(
        Line2D([0], [0], color='#4A90E2', linewidth=1.5, label='River')
    )

# 地级市边界图例
legend_elements.append(
    Line2D([0], [0], color='black', linewidth=1.0, label='City Boundary')
)

# 省级边界图例
legend_elements.append(
    Line2D([0], [0], color='#DDA0DD', linewidth=2.5, label='Province Boundary')
)

# # 地区边界图例（深紫色+浅紫色）
# legend_elements.append(
#     Line2D([0], [0], color='#9370DB', linewidth=3.0, label='Region Boundary (Deep Purple)')
# )
# legend_elements.append(
#     Line2D([0], [0], color='#DDA0DD', linewidth=4.0, label='Region Boundary (Light Purple)')
# )

# 添加图例到地图（无边框）
ax.legend(
    handles=legend_elements,
    loc='upper right',
    fontsize=12,
    frameon=False,  # 无边框
    fancybox=False,
    shadow=False
)

# 保存图片（无边框）
plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches=None,
    pad_inches=0,
    transparent=True,
    facecolor='none',
    edgecolor='none'
)

print(f"\n地图已保存为: {OUTPUT_PNG}")
print("完成！")
