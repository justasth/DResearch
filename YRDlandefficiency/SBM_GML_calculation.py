import pandas as pd
import numpy as np
import os
from scipy import optimize

data = pd.read_excel("raw_data.xlsx").values
# 参数设置
x_num, y_g_num, y_b_num= 1, 2, 1
period, DMU_num =4, 41
rtsc = 0   # 0 for CRS, 1 for VRS
rtsv = 1   # 0 for CRS, 1 for VRS
if data.shape[1] != x_num + y_g_num + y_b_num + 2:
    print("Data dimension error!!!")

else:
    # 计算全局"G_"SBM效率和GML指数
    # 收集打印结果，便于导出 CSV
    global_eff_rows = []
    gml_rows = []
    
    x_all = data[:, 2:2 + x_num]#所有行 + 第 2 列到2+x_num列，即所有 DMU 的投入指标数据
    y_g_all = data[:, 2 + x_num:2 + x_num + y_g_num]
    y_b_all = data[:, 2 + x_num + y_g_num:]
    SBM_global_eff_c = SBM_model(x_all.T, y_g_all.T, y_b_all.T, rtsc)#####计算CRS指数E，全局可比
    SBM_global_eff_v = SBM_model(x_all.T, y_g_all.T, y_b_all.T, rtsv)#####计算VRS指数PTE，全局可比
    SE_index = SBM_global_eff_c[:] / SBM_global_eff_v[:]#####计算SE指数，全局可比
    print("\n=== 全局CRS/VRS效率与规模报酬阶段 ===")
    print(f"{'ID':<10} {'Year':<10} {'E(CRS)':<12} {'PTE(VRS)':<12} {'SE':<12} {'Stage':<6}")
    print("-" * 60)
    for i in range(len(SE_index)):
        if SE_index[i] > 1:
            STAGE = "IRS"  # 规模报酬递增：扩大生产规模可提升效率
        elif SE_index[i] == 1:
            STAGE = "-"    # 规模报酬不变：生产规模处于最优状态
        else:
            STAGE = "DRS"  # 规模报酬递减：扩大生产规模会降低效率
        print(f"{data[i, 0]:<10} {data[i, 1]:<10} {SBM_global_eff_c[i]:<12.6f} {SBM_global_eff_v[i]:<12.6f} {SE_index[i]:<12.6f} {STAGE:<6}")
        # 记录全局效率结果
        global_eff_rows.append({
            "ID": data[i, 0],
            "Year": data[i, 1],
            "E_CRS": SBM_global_eff_c[i],
            "PTE_VRS": SBM_global_eff_v[i],
            "SE": SE_index[i],
            "Stage": STAGE
        })
    
    # 切分全局效率值，得到每个年份的Eg（4x41数组）
    Eg_1990_v = SBM_global_eff_v[0:DMU_num]  # 1990年全局VRS效率（41个城市）
    Eg_2000_v = SBM_global_eff_v[DMU_num:DMU_num*2]  # 2000年全局VRS效率（41个城市）
    Eg_2010_v = SBM_global_eff_v[DMU_num*2:DMU_num*3]  # 2010年全局VRS效率（41个城市）
    Eg_2020_v = SBM_global_eff_v[DMU_num*3:DMU_num*4]  # 2020年全局VRS效率（41个城市）
    
    Eg_1990_c = SBM_global_eff_c[0:DMU_num]  # 1990年全局CRS效率（41个城市）
    Eg_2000_c = SBM_global_eff_c[DMU_num:DMU_num*2]  # 2000年全局CRS效率（41个城市）
    Eg_2010_c = SBM_global_eff_c[DMU_num*2:DMU_num*3]  # 2010年全局CRS效率（41个城市）
    Eg_2020_c = SBM_global_eff_c[DMU_num*3:DMU_num*4]  # 2020年全局CRS效率（41个城市）
    
    # 切片计算1990的VRS效率
    x_all_1990 = data[:DMU_num, 2:2 + x_num]
    y_g_all_1990 = data[:DMU_num, 2 + x_num:2 + x_num + y_g_num]
    y_b_all_1990 = data[:DMU_num, 2 + x_num + y_g_num:]
    Et1_1990_v = SBM_model(x_all_1990.T, y_g_all_1990.T, y_b_all_1990.T, rtsv)
    Et1_1990_c = SBM_model(x_all_1990.T, y_g_all_1990.T, y_b_all_1990.T, rtsc)
    
    # 切片计算2000的VRS效率
    x_all_2000 = data[DMU_num:DMU_num*2, 2:2 + x_num]
    y_g_all_2000 = data[DMU_num:DMU_num*2, 2 + x_num:2 + x_num + y_g_num]
    y_b_all_2000 = data[DMU_num:DMU_num*2, 2 + x_num + y_g_num:]
    Et2_2000_v = SBM_model(x_all_2000.T, y_g_all_2000.T, y_b_all_2000.T, rtsv)
    Et2_2000_c = SBM_model(x_all_2000.T, y_g_all_2000.T, y_b_all_2000.T, rtsc)
    
    # 切片计算2010的VRS效率
    x_all_2010 = data[DMU_num*2:DMU_num*3, 2:2 + x_num]
    y_g_all_2010 = data[DMU_num*2:DMU_num*3, 2 + x_num:2 + x_num + y_g_num]
    y_b_all_2010 = data[DMU_num*2:DMU_num*3, 2 + x_num + y_g_num:]
    Et3_2010_v = SBM_model(x_all_2010.T, y_g_all_2010.T, y_b_all_2010.T, rtsv)
    Et3_2010_c = SBM_model(x_all_2010.T, y_g_all_2010.T, y_b_all_2010.T, rtsc)
    
    # 切片计算2020的VRS效率
    x_all_2020 = data[DMU_num*3:DMU_num*4, 2:2 + x_num]
    y_g_all_2020 = data[DMU_num*3:DMU_num*4, 2 + x_num:2 + x_num + y_g_num]
    y_b_all_2020 = data[DMU_num*3:DMU_num*4, 2 + x_num + y_g_num:]
    Et4_2020_v = SBM_model(x_all_2020.T, y_g_all_2020.T, y_b_all_2020.T, rtsv)
    Et4_2020_c = SBM_model(x_all_2020.T, y_g_all_2020.T, y_b_all_2020.T, rtsc)
    
    # 多期 GML 计算（1990-2000、2000-2010、2010-2020），用一个循环处理
    intervals = [
        ("1990-2000", Eg_1990_c, Eg_2000_c, Eg_1990_v, Eg_2000_v, Et1_1990_c, Et1_1990_v, Et2_2000_c, Et2_2000_v),
        ("2000-2010", Eg_2000_c, Eg_2010_c, Eg_2000_v, Eg_2010_v, Et2_2000_c, Et2_2000_v, Et3_2010_c, Et3_2010_v),
        ("2010-2020", Eg_2010_c, Eg_2020_c, Eg_2010_v, Eg_2020_v, Et3_2010_c, Et3_2010_v, Et4_2020_c, Et4_2020_v),
    ]

    for label, Eg_t, Eg_tp1, Eg_t_v, Eg_tp1_v, Et_t_c, Et_t_v, Et_tp1_c, Et_tp1_v in intervals:
        print(f"\n=== {label} GML指数计算 ===")
        print(f"{'ID':<10} {'Year':<10} {'GML':<12} {'TC':<12} {'EC':<12} {'PEC':<12} {'SEC':<12} {'M(方法一)':<12} {'M(方法二)':<12} {'GML-M差异':<12}")
        print("-" * 140)
        base_year = int(label.split("-")[0])

        for i in range(DMU_num):
            # 全局效率比值 GML
            GML_t_tp1 = Eg_tp1[i] / Eg_t[i]
            GPTEC_t_tp1 = Eg_tp1_v[i] / Eg_t_v[i]
            GSEC_t_tp1 = GML_t_tp1 / GPTEC_t_tp1

            # 局部前沿面效率比值（EC、PEC、SEC）
            Et_Pt_t = Et_t_c[i] / Et_t_v[i]
            Et_Pt_tp1 = Et_tp1_c[i] / Et_tp1_v[i]
            EC_t_tp1 = Et_Pt_tp1 / Et_Pt_t
            PEC_t_tp1 = Et_tp1_v[i] / Et_t_v[i]
            SEC_t_tp1 = EC_t_tp1 / PEC_t_tp1

            # 全局技术进步 TC = GML / EC
            TC_t_tp1 = GML_t_tp1 / EC_t_tp1

            M_method1 = TC_t_tp1 * EC_t_tp1
            M_method2 = TC_t_tp1 * PEC_t_tp1 * SEC_t_tp1

            diff = abs(M_method1 - M_method2)
            if diff > 1e-6:
                print(f"警告：城市{i}两种方法结果不一致，差异={diff:.10f}")

            city_id = data[i + 0, 0]  # 同一城市在各期位置一致（按顺序排列）
            year = base_year  # 起始年份
            diff_gml_m_display = abs(GML_t_tp1 - M_method1)
            print(f"{city_id:<10} {year:<10} {GML_t_tp1:<12.6f} {TC_t_tp1:<12.6f} {EC_t_tp1:<12.6f} "
                  f"{PEC_t_tp1:<12.6f} {SEC_t_tp1:<12.6f} {M_method1:<12.6f} {M_method2:<12.6f} {diff_gml_m_display:<12.10f}")

            gml_rows.append({
                "Interval": label,
                "ID": city_id,
                "Year_start": year,
                "GML": GML_t_tp1,
                "TC": TC_t_tp1,
                "EC": EC_t_tp1,
                "PEC": PEC_t_tp1,
                "SEC": SEC_t_tp1,
                "M_method1": M_method1,
                "M_method2": M_method2,
                "GML_minus_M": diff_gml_m_display
            })

    # 导出CSV结果
    os.makedirs("Results", exist_ok=True)
    pd.DataFrame(global_eff_rows).to_csv("Results/global_efficiency.csv", index=False)
    pd.DataFrame(gml_rows).to_csv("Results/gml_1990_2000.csv", index=False)
    
def SBM_model(x, y_g, y_b, rts):
    theta = []
    s1, s2 = y_g.shape[0], y_b.shape[0]
    m, n = x.shape
    for i in range(n):
        f = np.concatenate([np.zeros(n), -1/(m*x[:, i]),
                        np.zeros(s1+s2), np.array([1])])
        Aeq1 = np.hstack([x,
                        np.identity(m),
                        np.zeros((m, s1+s2)),
                        -x[:, i, None]])
        Aeq2 = np.hstack([y_g,
                        np.zeros((s1, m)),
                        -np.identity(s1),
                        np.zeros((s1, s2)),
                        -y_g[:, i, None]])
        Aeq3 = np.hstack([y_b,
                        np.zeros((s2, m)),
                        np.zeros((s2, s1)),
                        np.identity(s2),
                        -y_b[:, i, None]])
        Aeq4 = np.hstack([np.zeros(n),
                        np.zeros(m),
                        1/((s1+s2)*(y_g[:, i])),
                        1/((s1+s2)*(y_b[:, i])),
                        np.array([1])]).reshape(1, -1)
        Aeq5 = np.hstack([np.ones(n),
                       np.zeros((m+s1+s2)),
                       np.array([-1])]).reshape(1,-1)

        if rts == 0:
            Aeq = np.vstack([Aeq1, Aeq2, Aeq3, Aeq4])
            beq = np.concatenate([np.zeros(m+s1+s2),np.array([1])])
        elif rts == 1:
            Aeq = np.vstack([Aeq1, Aeq2, Aeq3, Aeq4, Aeq5])
            beq = np.concatenate([np.zeros(m+s1+s2), np.array([1, 0])])

        bounds = tuple([(0, None) for t in range(n+s1+s2+m+1)])
        res = optimize.linprog(c=f, A_eq=Aeq, b_eq=beq, bounds=bounds)
        theta.append(res.fun)
    theta = np.array(theta)
    return theta    
