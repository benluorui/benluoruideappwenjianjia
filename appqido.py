import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# 缓存数据加载函数（使用streamlit缓存优化性能）
@st.cache_data
def load_and_process_data(file_path='uber.csv', sample_size=5000):
    """加载并处理Uber车费数据集"""
    # 1. 加载数据
    df = pd.read_csv(file_path)
    
    # 2. 数据基本信息展示
    st.subheader("1. 数据集基本信息")
    st.write(f"原始数据集形状: {df.shape[0]}行, {df.shape[1]}列")
    st.write(f"采样后数据集形状: {sample_size}行, {df.shape[1]}列")
    
    # 3. 数据采样（确保不超过原始数据量）
    sample_size = min(sample_size, df.shape[0])
    df_sampled = df.sample(sample_size, random_state=42)  # 固定随机种子确保可复现
    
    # 4. 数据清洗
    st.subheader("2. 数据清洗")
    
    # 4.1 缺失值处理
    missing_values = df_sampled.isnull().sum()
    st.write("缺失值统计:")
    st.dataframe(missing_values[missing_values > 0].to_frame(name="缺失值数量"))
    
    # 移除包含缺失值的行（对于这个数据集更合理）
    df_cleaned = df_sampled.dropna()
    st.write(f"清洗后数据量: {df_cleaned.shape[0]}行")
    
    # 4.2 数据类型转换 + 新增：提取年、月、日、星期字段
    # 将日期列转换为datetime类型（添加errors='coerce'，避免异常时间格式报错）
    df_cleaned['pickup_datetime'] = pd.to_datetime(df_cleaned['pickup_datetime'], errors='coerce')
    
    # 新增1：提取年份（如2023）
    df_cleaned['year'] = df_cleaned['pickup_datetime'].dt.year
    # 新增2：提取月份（1-12，如3代表3月）
    df_cleaned['month'] = df_cleaned['pickup_datetime'].dt.month
    # 新增3：提取日期（1-31，如15代表当月15日）
    df_cleaned['day'] = df_cleaned['pickup_datetime'].dt.day
    # 新增4：提取星期（0=周一，6=周日，方便后续按星期分析）
    df_cleaned['weekday'] = df_cleaned['pickup_datetime'].dt.dayofweek
    df_cleaned['quarter'] = df_cleaned['pickup_datetime'].dt.quarter  # 季度（1-4）
    df_cleaned['hour'] = df_cleaned['pickup_datetime'].dt.hour  # 时段（小时，0-23）
   


    
    # 过滤因时间格式异常导致的空值（确保新增字段无无效数据）
    df_cleaned = df_cleaned.dropna(subset=['pickup_datetime', 'year', 'month', 'day', 'weekday'])
    
    # 4.3 异常值处理（基于业务逻辑）
    # 移除 fare_amount 为负数或0的记录
    df_cleaned = df_cleaned[df_cleaned['fare_amount'] > 0]
    # 移除乘客数量为0的记录
    df_cleaned = df_cleaned[df_cleaned['passenger_count'] > 0]
    # 移除经纬度异常值（基于纽约大致经纬度范围）
    df_cleaned = df_cleaned[
        (df_cleaned['pickup_latitude'].between(40.4, 41.0)) &
        (df_cleaned['pickup_longitude'].between(-74.3, -73.7)) &
        (df_cleaned['dropoff_latitude'].between(40.4, 41.0)) &
        (df_cleaned['dropoff_longitude'].between(-74.3, -73.7))
    ]
    
    st.write(f"异常值处理后数据量: {df_cleaned.shape[0]}行")
    
    # 新增：展示提取的年月日字段示例（方便验证是否成功）
    st.write("提取的时间字段示例（前5行）:")
    st.dataframe(df_cleaned[['pickup_datetime', 'year', 'month', 'day', 'weekday']].head())
    df['month'] = df['pickup_datetime'].dt.month


    df['month'] = df['pickup_datetime'].dt.month
# 2. 手动映射中文月份（无任何locale依赖）
    month_map = {1:'1月',2:'2月',3:'3月',4:'4月',5:'5月',6:'6月',
             7:'7月',8:'8月',9:'9月',10:'10月',11:'11月',12:'12月'}
    df['month_name'] = df['month'].map(month_map)
# 2. 手动映射中文月份（无任何locale依赖）
  
    return df_cleaned

# 数据探索函数（必须放在 load_and_process_data() 之后、main() 之前）
def explore_data(df):
    """详细探索数据集特征"""
    st.subheader("3. 数据集详细描述")
    
    # 3.1 数据概览
    st.write("数据前5行:")
    st.dataframe(df.head())
    
    # 3.2 数值特征统计描述
    st.write("数值特征统计描述:")
    st.dataframe(df.describe())
    

    

    st.subheader("金额分布图")
    # 4.1 车费金额分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df['fare_amount'], kde=True, bins=30)
    plt.title('车费金额分布')
    plt.xlabel('车费金额($)')
    st.pyplot(plt)
    plt.close()
    st.subheader("乘客数量分布图")

    plt.figure(figsize=(10, 6))
    sns.countplot(x='passenger_count', data=df)
    plt.title('乘客数量分布')
    plt.xlabel('乘客数量')
    st.pyplot(plt)
    plt.close()
    st.subheader("时间分布图")
    # 4.3 时间特征分析
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek  # 0=周一, 6=周日
    
    plt.figure(figsize=(12, 5))
    sns.boxplot(x='hour', y='fare_amount', data=df)
    plt.title('不同时段的车费分布')
    plt.xlabel('小时')
    plt.ylabel('车费金额($)')
    st.pyplot(plt)
    plt.close()


    st.subheader("地理分布图")
    # 4.4 地理分布简单展示（已解决列名问题）
    st.write("4.4 上下车地点经纬度分布")
    sample_data = df[['pickup_latitude', 'pickup_longitude']].sample(1000)
    sample_data = sample_data.rename(columns={
        'pickup_latitude': 'latitude',
        'pickup_longitude': 'longitude'
    })
    st.map(sample_data)
# 商业价值分析
def business_value_analysis(df):
    """分析数据集的商业价值"""
    st.subheader("5. 商业价值分析")


    # 5.1 时段与车费总和分析（折线图）
    # 按小时分组计算车费总和
    hourly_total_fare = df.groupby('hour')['fare_amount'].sum().reset_index()
    # 绘制时段-车费总和折线图
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='hour', y='fare_amount', data=hourly_total_fare, marker='o')
    plt.title('不同时段的车费总和趋势')
    plt.xlabel('小时（24小时制）')
    plt.ylabel('车费总和($)')
    plt.xticks(range(0, 24, 2))  # 每2小时显示一个刻度，避免拥挤
    st.pyplot(plt)
    plt.close()
    # 补充关键结论
    max_total_hour = hourly_total_fare.loc[hourly_total_fare['fare_amount'].idxmax()]
    st.write(f"• 车费总和最高的时段: {int(max_total_hour['hour'])}点，总车费: ${max_total_hour['fare_amount']:.2f}")
    
    # 5.2 地区与乘坐人数分析（热力图）
    # 经纬度分组（保留1位小数，避免坐标过于分散）
    df['pickup_lat_round'] = df['pickup_latitude'].round(1)
    df['pickup_lon_round'] = df['pickup_longitude'].round(1)
    # 按经纬度聚合，统计各地区总乘坐人数
    area_passenger = df.groupby(['pickup_lat_round', 'pickup_lon_round'])['passenger_count'].sum().reset_index()
    area_passenger = area_passenger[area_passenger['passenger_count'] > 0]  # 过滤无数据区域
    
    # 绘制地区-乘坐人数热力图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        x=area_passenger['pickup_lon_round'], 
        y=area_passenger['pickup_lat_round'],
        s=area_passenger['passenger_count']/10,  # 调整点大小比例
        c=area_passenger['passenger_count'], 
        cmap='YlOrRd',  # 暖色调，人数越多颜色越深
        alpha=0.7
    )
    plt.colorbar(scatter, label='总乘坐人数')  # 添加颜色解释条
    plt.title('不同地区的乘坐人数分布')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    st.pyplot(plt)
    plt.close()
    # 补充关键结论
    top_area = area_passenger.loc[area_passenger['passenger_count'].idxmax()]
    st.write(f"• 乘坐人数最多的区域: 经度{top_area['pickup_lon_round']}，纬度{top_area['pickup_lat_round']}，总人数: {int(top_area['passenger_count'])}人")
  

    st.write("### 5.0 各年总收入统计")
    if 'year' not in df.columns:
        st.error("数据中缺少'year'字段，请先确保load_and_process_data()正确提取了年份！")
        return
    
    # 推荐使用方案1：直接指定函数名（兼容性最佳）
    yearly_total_income = df.groupby('year')['fare_amount'].agg(
        年度总收入_美元='sum',
        年度订单总数='count'
    ).reset_index()
    
    # 数据格式化
    yearly_total_income['year'] = yearly_total_income['year'].astype(int)
    yearly_total_income['年度总收入_美元'] = yearly_total_income['年度总收入_美元'].round(2)
    
    # 展示表格
    st.dataframe(
        yearly_total_income.rename(columns={
            'year': '年份',
            '年度总收入_美元': '年度总收入（$）',
            '年度订单总数': '年度订单总数'
        }),
        use_container_width=True
    )


    plt.figure(figsize=(10, 6))  # 图表尺寸，与原代码保持一致
    # 绘制总收入折线（主趋势）
    sns.lineplot(
        data=yearly_total_income,
        x='year', 
        y='年度总收入_美元', 
        marker='o',  # 添加圆形数据点，便于查看具体数值
        linewidth=2.5,  # 线条加粗，更醒目
        color='#2E86AB',  # 蓝色系，与原代码可视化风格统一
        label='年度总收入'
    )
    plt.title('Uber各年总收入趋势图', fontsize=14, pad=20)  # 标题及间距
    plt.xlabel('年份', fontsize=12)  # X轴标签
    plt.ylabel('总收入（$）', fontsize=12)  # Y轴标签
    plt.xticks(yearly_total_income['year'], rotation=0)  # 年份横向显示，避免重叠
    plt.grid(alpha=0.3)  # 浅色网格，提升可读性
    plt.legend(fontsize=11)  # 图例
    
    # 4. 在Streamlit中展示图表
    st.pyplot(plt)
    plt.close()  # 关闭图表，避免内存占用
    
    # 5. 补充趋势结论（增强分析价值）
    if len(yearly_total_income) >= 2:
        income_change = yearly_total_income['年度总收入_美元'].iloc[-1] - yearly_total_income['年度总收入_美元'].iloc[0]
        change_rate = (income_change / yearly_total_income['年度总收入_美元'].iloc[0]) * 100
        st.write(f"• 趋势总结：从{yearly_total_income['year'].iloc[0]}年到{yearly_total_income['year'].iloc[-1]}年，总收入{'增长' if income_change > 0 else '下降'}${abs(income_change):.2f}，变化率{change_rate:.2f}%")
    


    st.write("### 5.1 不同年份和季度的乘坐人数变化")
    # 1. 分组计算
    year_quarter_passenger = df.groupby(['year', 'quarter'])['passenger_count'].agg(
        总乘坐人数='sum',
        季度订单数='count'
    ).reset_index()
    
    # 2. 格式化并添加季度名称
    year_quarter_passenger['年份'] = year_quarter_passenger['year'].astype(int)
    year_quarter_passenger['季度'] = year_quarter_passenger['quarter'].map({
        1: 'Q1（1-3月）',
        2: 'Q2（4-6月）',
        3: 'Q3（7-9月）',
        4: 'Q4（10-12月）'
    })
    
    # 3. 排序时使用新列名（修复KeyError的核心）
    # 原错误：by=['年份', 'quarter'] → 改为 by=['年份', 'quarter'] 或 by=['年份', '季度']
    # 推荐：按原始quarter数字排序（确保Q1-Q4顺序正确）
    year_quarter_passenger = year_quarter_passenger.sort_values(by=['年份', 'quarter'])
    
    # 4. 展示表格（只显示需要的列，隐藏原始quarter）
    st.dataframe(
        year_quarter_passenger[['年份', '季度', '总乘坐人数', '季度订单数']],
        use_container_width=True
    )
    
    # 5.2 各个时段的乘坐人数变化表（不变，若有问题可参照上述逻辑修复）
    st.write("### 5.2 各个时段（小时）的乘坐人数变化")
    hourly_passenger = df.groupby('hour')['passenger_count'].agg(
        总乘坐人数='sum',
        时段订单数='count'
    ).reset_index()
    hourly_passenger['小时（24小时制）'] = hourly_passenger['hour'].astype(int)
    hourly_passenger['时段描述'] = hourly_passenger['hour'].apply(lambda x: f"{x}点-{x+1}点")
    hourly_passenger = hourly_passenger.sort_values(by='hour')
    st.dataframe(
        hourly_passenger[['小时（24小时制）', '时段描述', '总乘坐人数', '时段订单数']],
        use_container_width=True
    )
    
    
   



    # 5.3 潜在商业应用

    st.write("""
    • 商业应用1: 动态定价策略 - 基于高峰时段调整价格，最大化收益  
    • 商业应用2: 车辆调度优化 - 根据上下车热点区域合理分配车辆  
    • 商业应用3: 市场细分 - 针对不同乘客数量的群体设计差异化服务  
    • 商业应用4: 路线规划优化 - 分析高频路线，
    """)

# 主函数
def main():
    st.title("Uber的没有实际意义的分析App由不知名男子使用AI开发，请仔细甄别")
    
    # 加载并处理数据
    df = load_and_process_data()
    
    # 数据探索
    explore_data(df)
    
    # 商业价值分析
    business_value_analysis(df)
    
    # 提供清洗后的数据集下载
    st.subheader("6. 数据下载")
    csv = df.to_csv(index=False)
    st.download_button(
        label="下载处理后的CSV文件",
        data=csv,
        file_name="processed_uber_fares.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv('processed_uber_fares.csv')

# 设置页面标题
st.title('Uber 车费数据集分析应用')

# 显示数据基本信息
st.header('数据基本信息')
st.write(df.info())


st.header('一天中的不同时段如何影响车费金额？')

# 添加加载动画（数据处理时显示）
with st.spinner('正在处理时间数据...'):
    # 转换 pickup_datetime 为日期时间类型（添加容错处理）
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    
    # 提取小时信息（过滤无效时间）
    df = df.dropna(subset=['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['hour'] = df['hour'].astype(int)

# 数据加载完成后显示气球特效
st.balloons()  # 新增：释放彩色气球

# 筛选组件 - 美化滑块样式
st.subheader('选择分析的小时范围')
hour_min = int(df['hour'].min())
hour_max = int(df['hour'].max())
selected_hour = st.slider(
    '拖动滑块选择时段范围',
    min_value=hour_min,
    max_value=hour_max,
    value=(hour_min, hour_max),
    format='%d点'  # 显示为“X点”格式
)

# 过滤数据（添加数据量反馈）
filtered_df_hour = df[(df['hour'] >= selected_hour[0]) & (df['hour'] <= selected_hour[1])]
st.info(f'已筛选出 {selected_hour[0]}点 至 {selected_hour[1]}点 的数据，共 {len(filtered_df_hour)} 条记录')

# 分析不同时段的平均车费金额（添加计算成功提示）
if not filtered_df_hour.empty:
    hourly_avg_fare = filtered_df_hour.groupby('hour')['fare_amount'].mean().reset_index()
    st.success('数据计算完成！')
    
    # 绘制折线图（添加动态高亮特效）
    plt.figure(figsize=(10, 6))
    
    # 基础折线
    sns.lineplot(
        x='hour', 
        y='fare_amount', 
        data=hourly_avg_fare, 
        marker='o',
        linewidth=2,
        color='#4285F4'  # 谷歌蓝
    )
    
    # 高亮峰值点（特效1：标记最高车费时段）
    peak_idx = hourly_avg_fare['fare_amount'].idxmax()
    peak_hour = hourly_avg_fare.loc[peak_idx]
    plt.scatter(
        peak_hour['hour'], 
        peak_hour['fare_amount'], 
        color='red', 
        s=100,  # 点大小
        zorder=5,  # 置于顶层
        label=f'峰值：{peak_hour["hour"]}点（${peak_hour["fare_amount"]:.2f}）'
    )
    
    # 图表美化（特效2：动态标题和网格）
    plt.title(
        f'{selected_hour[0]}点至{selected_hour[1]}点的平均车费趋势',
        fontsize=14,
        pad=20
    )
    plt.xlabel('小时（24小时制）', fontsize=12)
    plt.ylabel('平均车费金额（$）', fontsize=12)
    plt.xticks(range(selected_hour[0], selected_hour[1]+1), rotation=0)  # 只显示选中范围的小时
    plt.grid(alpha=0.3, linestyle='--')  # 虚线网格
    plt.legend()
    
    # 显示图表（特效3：添加边框阴影）
    st.pyplot(plt)
    
    # 显示分析结果（特效4：高亮峰值行）
    st.subheader('时段平均车费详情')
    # 为峰值行添加高亮样式
    def highlight_peak(row):
        if row['hour'] == peak_hour['hour']:
            return ['background-color: #FFCCCC'] * len(row)  # 浅红色背景
        else:
            return [''] * len(row)
    styled_df = hourly_avg_fare.style.apply(highlight_peak, axis=1).format({
        'fare_amount': '${:.2f}'  # 格式化车费为美元格式
    })
    st.dataframe(styled_df, use_container_width=True)

else:
    st.warning('所选时段内无数据，请调整范围！')






st.header('问题：各月份与乘坐人数之间的关系')

# 1. 数据加载与字段提取（确保月份字段存在）
with st.spinner('正在处理月份数据...'):
    # 强制提取月份字段（避免KeyError）
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df = df.dropna(subset=['pickup_datetime'])
    df['year'] = df['pickup_datetime'].dt.year.astype(int)  # 保留年份用于筛选
    df['month'] = df['pickup_datetime'].dt.month.astype(int)  # 提取月份（1-12）
    df['month_name'] = df['pickup_datetime'].dt.month_name(locale='zh_CN')  # 月份中文名（1→1月）
    
    # 获取有效年份范围
    valid_years = sorted(df['year'].unique())
    st.balloons()  # 加载完成释放气球特效

# 2. 年份筛选组件（可选分析单年或多年）
st.subheader('选择分析的年份')
selected_year = st.selectbox(
    '选择特定年份查看月度趋势（选"所有年份"则合并统计）',
    options=['所有年份'] + [str(year) for year in valid_years],
    index=0
)

# 3. 数据过滤与聚合
# 按选择的年份过滤数据
if selected_year != '所有年份':
    filtered_df = df[df['year'] == int(selected_year)]
    analysis_title = f'{selected_year}年各月份乘坐人数趋势'
else:
    filtered_df = df  # 合并所有年份数据
    analysis_title = '所有年份合并：各月份乘坐人数趋势'

# 反馈数据量
st.info(f'当前分析数据范围：{analysis_title}，共 {len(filtered_df):,} 条记录')

# 4. 核心分析：按月份聚合乘坐人数
if not filtered_df.empty:
    # 按月份分组（按1-12月排序）
    monthly_passenger = filtered_df.groupby(['month', 'month_name'])['passenger_count'].sum().reset_index()
    monthly_passenger.rename(columns={'passenger_count': '总乘坐人数'}, inplace=True)
    # 按月份数字排序（确保1-12月顺序）
    monthly_passenger = monthly_passenger.sort_values('month').reset_index(drop=True)
    
    st.success('月度乘坐人数数据计算完成！')

    # 5. 绘制月度趋势图（带峰值高亮）
    plt.figure(figsize=(12, 6))
    
    # 基础折线：月份 vs 总乘坐人数
    sns.lineplot(
        x='month_name', 
        y='总乘坐人数', 
        data=monthly_passenger,
        marker='o',  # 圆形标记点
        linewidth=2.5,
        color='#2E86AB'  # 统一蓝色系
    )
    
    # 高亮乘坐人数最高的月份（峰值特效）
    peak_month = monthly_passenger.loc[monthly_passenger['总乘坐人数'].idxmax()]
    plt.scatter(
        peak_month['month_name'], 
        peak_month['总乘坐人数'],
        color='red', 
        s=150,  # 放大峰值点
        zorder=10,  # 置于顶层
        edgecolors='white',
        linewidth=2,
        label=f'峰值：{peak_month["month_name"]}（{peak_month["总乘坐人数"]:,}人）'
    )
    
    # 图表美化（适配月份特征）
    plt.title(analysis_title, fontsize=14, pad=20)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('总乘坐人数', fontsize=12)
    plt.xticks(rotation=45)  # 月份名横向排列，避免重叠
    plt.grid(alpha=0.3, linestyle='--')  # 浅色虚线网格
    plt.legend(loc='upper right')  # 图例放右上角
    plt.tight_layout()  # 自动调整布局（避免标签被截断）
    st.pyplot(plt)
    st.balloons()
    st.snow() 
    st.balloons()
    st.snow() 
    st.balloons()
    st.snow() 
   