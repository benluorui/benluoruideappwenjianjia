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
    
    # 移除包含缺失值的行
    df_cleaned = df_sampled.dropna()
    st.write(f"清洗后数据量: {df_cleaned.shape[0]}行")
    
    # 4.2 数据类型转换 + 提取时间字段
    df_cleaned['pickup_datetime'] = pd.to_datetime(df_cleaned['pickup_datetime'], errors='coerce')
    
    # 提取年、月、日、星期等字段
    df_cleaned['year'] = df_cleaned['pickup_datetime'].dt.year
    df_cleaned['month'] = df_cleaned['pickup_datetime'].dt.month
    df_cleaned['day'] = df_cleaned['pickup_datetime'].dt.day
    df_cleaned['weekday'] = df_cleaned['pickup_datetime'].dt.dayofweek
    df_cleaned['quarter'] = df_cleaned['pickup_datetime'].dt.quarter
    df_cleaned['hour'] = df_cleaned['pickup_datetime'].dt.hour  # 时段（小时，0-23）
    
    # 过滤无效时间数据
    df_cleaned = df_cleaned.dropna(subset=['pickup_datetime', 'year', 'month', 'day', 'weekday'])
    
    # 4.3 异常值处理
    df_cleaned = df_cleaned[df_cleaned['fare_amount'] > 0]  # 车费为正
    df_cleaned = df_cleaned[df_cleaned['passenger_count'] > 0]  # 乘客数为正
    # 经纬度范围过滤（纽约区域）
    df_cleaned = df_cleaned[
        (df_cleaned['pickup_latitude'].between(40.4, 41.0)) &
        (df_cleaned['pickup_longitude'].between(-74.3, -73.7)) &
        (df_cleaned['dropoff_latitude'].between(40.4, 41.0)) &
        (df_cleaned['dropoff_longitude'].between(-74.3, -73.7))
    ]
    
    st.write(f"异常值处理后数据量: {df_cleaned.shape[0]}行")
    
    # 展示时间字段示例
    st.write("提取的时间字段示例（前5行）:")
    st.dataframe(df_cleaned[['pickup_datetime', 'year', 'month', 'day', 'weekday']].head())
    
    return df_cleaned

# 数据探索函数
def explore_data(df):
    """详细探索数据集特征"""
    st.subheader("3. 数据集详细描述")
    
    # 3.1 数据概览
    st.write("数据前5行:")
    st.dataframe(df.head())
    
    # 3.2 数值特征统计描述
    st.write("数值特征统计描述:")
    st.dataframe(df.describe())
    
    # 4.1 车费金额分布
    st.subheader("金额分布图")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['fare_amount'], kde=True, bins=30)
    plt.title('车费金额分布')
    plt.xlabel('车费金额($)')
    st.pyplot(plt)
    plt.close()
    
    # 4.2 乘客数量分布
    st.subheader("乘客数量分布图")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='passenger_count', data=df)
    plt.title('乘客数量分布')
    plt.xlabel('乘客数量')
    st.pyplot(plt)
    plt.close()
    
    # 4.3 时段与车费分布
    st.subheader("时间分布图")
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    plt.figure(figsize=(12, 5))
    sns.boxplot(x='hour', y='fare_amount', data=df)
    plt.title('不同时段的车费分布')
    plt.xlabel('小时')
    plt.ylabel('车费金额($)')
    st.pyplot(plt)
    plt.close()
    
    # 4.4 地理分布
    st.subheader("地理分布图")
    st.write("4.4 上下车地点经纬度分布")
    sample_data = df[['pickup_latitude', 'pickup_longitude']].sample(1000)
    sample_data = sample_data.rename(columns={
        'pickup_latitude': 'latitude',
        'pickup_longitude': 'longitude'
    })
    st.map(sample_data)

# 商业价值分析函数
def business_value_analysis(df):
    """分析数据集的商业价值"""
    st.subheader("5. 商业价值分析")
    
    # 5.1 时段与车费总和分析
    hourly_total_fare = df.groupby('hour')['fare_amount'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='hour', y='fare_amount', data=hourly_total_fare, marker='o')
    plt.title('不同时段的车费总和趋势')
    plt.xlabel('小时（24小时制）')
    plt.ylabel('车费总和($)')
    plt.xticks(range(0, 24, 2))
    st.pyplot(plt)
    plt.close()
    max_total_hour = hourly_total_fare.loc[hourly_total_fare['fare_amount'].idxmax()]
    st.write(f"• 车费总和最高的时段: {int(max_total_hour['hour'])}点，总车费: ${max_total_hour['fare_amount']:.2f}")
    
    # 5.2 地区与乘坐人数分析
    df['pickup_lat_round'] = df['pickup_latitude'].round(1)
    df['pickup_lon_round'] = df['pickup_longitude'].round(1)
    area_passenger = df.groupby(['pickup_lat_round', 'pickup_lon_round'])['passenger_count'].sum().reset_index()
    area_passenger = area_passenger[area_passenger['passenger_count'] > 0]
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        x=area_passenger['pickup_lon_round'], 
        y=area_passenger['pickup_lat_round'],
        s=area_passenger['passenger_count']/10,
        c=area_passenger['passenger_count'], 
        cmap='YlOrRd',
        alpha=0.7
    )
    plt.colorbar(scatter, label='总乘坐人数')
    plt.title('不同地区的乘坐人数分布')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    st.pyplot(plt)
    plt.close()
    top_area = area_passenger.loc[area_passenger['passenger_count'].idxmax()]
    st.write(f"• 乘坐人数最多的区域: 经度{top_area['pickup_lon_round']}，纬度{top_area['pickup_lat_round']}，总人数: {int(top_area['passenger_count'])}人")
    
    # 5.0 各年总收入统计
    st.write("### 5.0 各年总收入统计")
    if 'year' not in df.columns:
        st.error("数据中缺少'year'字段！")
        return
    yearly_total_income = df.groupby('year')['fare_amount'].agg(
        年度总收入_美元='sum',
        年度订单总数='count'
    ).reset_index()
    yearly_total_income['year'] = yearly_total_income['year'].astype(int)
    yearly_total_income['年度总收入_美元'] = yearly_total_income['年度总收入_美元'].round(2)
    st.dataframe(
        yearly_total_income.rename(columns={
            'year': '年份',
            '年度总收入_美元': '年度总收入（$）',
            '年度订单总数': '年度订单总数'
        }),
        use_container_width=True
    )
    # 绘制年收入趋势图
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=yearly_total_income,
        x='year', 
        y='年度总收入_美元', 
        marker='o',
        linewidth=2.5,
        color='#2E86AB',
        label='年度总收入'
    )
    plt.title('Uber各年总收入趋势图', fontsize=14, pad=20)
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('总收入（$）', fontsize=12)
    plt.xticks(yearly_total_income['year'], rotation=0)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    st.pyplot(plt)
    plt.close()
    # 趋势结论
    if len(yearly_total_income) >= 2:
        income_change = yearly_total_income['年度总收入_美元'].iloc[-1] - yearly_total_income['年度总收入_美元'].iloc[0]
        change_rate = (income_change / yearly_total_income['年度总收入_美元'].iloc[0]) * 100
        st.write(f"• 趋势总结：从{yearly_total_income['year'].iloc[0]}年到{yearly_total_income['year'].iloc[-1]}年，总收入{'增长' if income_change > 0 else '下降'}${abs(income_change):.2f}，变化率{change_rate:.2f}%")
    
    # 5.1 不同年份和季度的乘坐人数变化
    st.write("### 5.1 不同年份和季度的乘坐人数变化")
    year_quarter_passenger = df.groupby(['year', 'quarter'])['passenger_count'].agg(
        总乘坐人数='sum',
        季度订单数='count'
    ).reset_index()
    year_quarter_passenger['年份'] = year_quarter_passenger['year'].astype(int)
    year_quarter_passenger['季度'] = year_quarter_passenger['quarter'].map({
        1: 'Q1（1-3月）', 2: 'Q2（4-6月）', 3: 'Q3（7-9月）', 4: 'Q4（10-12月）'
    })
    year_quarter_passenger = year_quarter_passenger.sort_values(by=['年份', 'quarter'])
    st.dataframe(
        year_quarter_passenger[['年份', '季度', '总乘坐人数', '季度订单数']],
        use_container_width=True
    )
    
    # 5.2 各个时段的乘坐人数变化
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
    • 商业应用4: 路线规划优化 - 分析高频路线，提升运营效率
    """)

# 第一互动插件：时段与车费金额关系
def hour_fare_analysis(df):
    st.header('互动分析1：一天中的不同时段如何影响车费金额？')
    
    # 数据处理（确保时间格式正确）
    with st.spinner('正在处理时间数据...'):
        # 转换并过滤时间字段
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
        df = df.dropna(subset=['pickup_datetime'])
        df['hour'] = df['pickup_datetime'].dt.hour.astype(int)  # 提取小时（0-23）
        st.balloons()  # 加载完成提示
    
    # 时段筛选滑块
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
    
    # 过滤数据并反馈
    filtered_df_hour = df[(df['hour'] >= selected_hour[0]) & (df['hour'] <= selected_hour[1])]
    st.info(f'已筛选出 {selected_hour[0]}点 至 {selected_hour[1]}点 的数据，共 {len(filtered_df_hour)} 条记录')
    
    # 分析并可视化
    if not filtered_df_hour.empty:
        hourly_avg_fare = filtered_df_hour.groupby('hour')['fare_amount'].mean().reset_index()
        st.success('数据计算完成！')
        
        # 绘制时段-平均车费折线图
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            x='hour', 
            y='fare_amount', 
            data=hourly_avg_fare, 
            marker='o',
            linewidth=2,
            color='#4285F4'
        )
        
        # 高亮峰值点
        peak_idx = hourly_avg_fare['fare_amount'].idxmax()
        peak_hour = hourly_avg_fare.loc[peak_idx]
        plt.scatter(
            peak_hour['hour'], 
            peak_hour['fare_amount'], 
            color='red', 
            s=100,
            zorder=5,
            label=f'峰值：{peak_hour["hour"]}点（${peak_hour["fare_amount"]:.2f}）'
        )
        
        # 图表美化
        plt.title(
            f'{selected_hour[0]}点至{selected_hour[1]}点的平均车费趋势',
            fontsize=14,
            pad=20
        )
        plt.xlabel('小时（24小时制）', fontsize=12)
        plt.ylabel('平均车费金额（$）', fontsize=12)
        plt.xticks(range(selected_hour[0], selected_hour[1]+1), rotation=0)
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend()
        st.pyplot(plt)
        plt.close()
        
        # 显示时段详情表格（高亮峰值）
        st.subheader('时段平均车费详情')
        def highlight_peak(row):
            if row['hour'] == peak_hour['hour']:
                return ['background-color: #FFCCCC'] * len(row)
            else:
                return [''] * len(row)
        styled_df = hourly_avg_fare.style.apply(highlight_peak, axis=1).format({
            'fare_amount': '${:.2f}'
        })
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.warning('所选时段内无数据，请调整范围！')

# 第二互动插件：热门路线与车费关系
def route_analysis(df):
    st.header('互动分析2：热门路线与车费关系')
    
    # 1. 数据预处理
    with st.spinner('正在加载路线数据...'):
        # 过滤有效经纬度
        df = df[
            (df['pickup_latitude'].between(40.4, 41.0)) &
            (df['pickup_longitude'].between(-74.3, -73.7)) &
            (df['dropoff_latitude'].between(40.4, 41.0)) &
            (df['dropoff_longitude'].between(-74.3, -73.7))
        ]
        
        # 简化经纬度并生成路线标识
        df['pickup_lat_round'] = df['pickup_latitude'].round(2)
        df['pickup_lon_round'] = df['pickup_longitude'].round(2)
        df['dropoff_lat_round'] = df['dropoff_latitude'].round(2)
        df['dropoff_lon_round'] = df['dropoff_longitude'].round(2)
        df['route'] = df.apply(
            lambda x: f"({x['pickup_lat_round']},{x['pickup_lon_round']})→({x['dropoff_lat_round']},{x['dropoff_lon_round']})",
            axis=1
        )
        st.balloons()
    
    # 2. 互动筛选组件
    col1, col2 = st.columns(2)
    with col1:
        min_orders = st.slider(
            '最低订单数量（筛选热门路线）',
            min_value=5, max_value=50, value=10,
            help='只显示订单数≥此值的路线'
        )
    with col2:
        # 显示名称与实际列名映射
        analysis_metric_display = st.selectbox(
            '选择分析指标',
            options=['平均车费($)', '总订单数', '总乘客数'],
            index=0
        )
    metric_mapping = {
        '平均车费($)': '平均车费',
        '总订单数': '总订单数',
        '总乘客数': '总乘客数'
    }
    analysis_metric = metric_mapping[analysis_metric_display]
    
    # 3. 数据聚合与过滤（避免NameError）
    required_cols = ['route', 'fare_amount', 'passenger_count', 
                    'pickup_lat_round', 'pickup_lon_round', 
                    'dropoff_lat_round', 'dropoff_lon_round']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"数据缺少必要字段，无法分析路线：{missing_cols}")
        route_stats = pd.DataFrame(columns=['route', '平均车费', '总订单数', '总乘客数', 
                                          '起点纬度', '起点经度', '终点纬度', '终点经度'])
    else:
        route_stats = df.groupby('route').agg(
            平均车费=('fare_amount', 'mean'),
            总订单数=('route', 'count'),
            总乘客数=('passenger_count', 'sum'),
            起点纬度=('pickup_lat_round', 'first'),
            起点经度=('pickup_lon_round', 'first'),
            终点纬度=('dropoff_lat_round', 'first'),
            终点经度=('dropoff_lon_round', 'first')
        ).reset_index()
    
    # 确保route_stats已定义
    if 'route_stats' not in locals():
        route_stats = pd.DataFrame(columns=['route', '平均车费', '总订单数', '总乘客数', 
                                          '起点纬度', '起点经度', '终点纬度', '终点经度'])
    
    # 筛选并排序
    filtered_routes = route_stats[route_stats['总订单数'] >= min_orders].sort_values(
        analysis_metric, ascending=False
    ).head(20)
    
    # 4. 互动可视化
    if not filtered_routes.empty:
        st.success(f'已筛选出 {len(filtered_routes)} 条热门路线（订单数≥{min_orders}）')
        
        # 地图展示起点分布
        st.subheader('热门路线起点分布')
        map_data = filtered_routes[['起点纬度', '起点经度']].rename(
            columns={'起点纬度': 'latitude', '起点经度': 'longitude'}
        )
        st.map(map_data, zoom=11)
        
        # 指标排行榜
        st.subheader(f'热门路线{analysis_metric_display}排行榜')
        plt.figure(figsize=(10, 8))
        if analysis_metric == '平均车费':
            sns.barplot(x='平均车费', y='route', data=filtered_routes, palette='coolwarm')
            plt.xlabel('平均车费($)')
        elif analysis_metric == '总订单数':
            sns.barplot(x='总订单数', y='route', data=filtered_routes, palette='viridis')
            plt.xlabel('总订单数')
        else:
            sns.barplot(x='总乘客数', y='route', data=filtered_routes, palette='magma')
            plt.xlabel('总乘客数')
        plt.ylabel('路线（起点→终点）')
        plt.title(f'热门路线{analysis_metric_display}TOP20', pad=20)
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        # 详细数据表格
        st.subheader('路线详细数据')
        display_cols = {
            'route': '路线',
            '平均车费': '平均车费($)',
            '总订单数': '总订单数',
            '总乘客数': '总乘客数'
        }
        st.dataframe(
            filtered_routes.rename(columns=display_cols)[display_cols.values()],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(f'无满足条件的路线，请降低订单量阈值（当前≥{min_orders}）')

# 主函数
def main():
    st.title("Uber数据分析App")
    
    # 加载并处理数据
    df = load_and_process_data()
    
    # 数据探索
    explore_data(df)
    
    # 商业价值分析
    business_value_analysis(df)
    
    # 互动分析模块（先时段-车费，再路线分析）
    hour_fare_analysis(df)
    route_analysis(df)
    
    # 数据下载
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








   




