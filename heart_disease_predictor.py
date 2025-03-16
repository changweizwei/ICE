import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('XGBoost.pkl')  # 加载训练好的XGBoost模型

# Streamlit UI
st.title("Wind Turbine Fault Predictor")  # 风力发电机故障预测器

# Sidebar for input options
st.sidebar.header("Input Sample Data")  # 侧边栏输入样本数据

# Yaw position input
yaw_position = st.sidebar.number_input("Yaw Position:", min_value=-10.0, max_value=10.0, value=0.0)  # 偏航位置输入框

# Environment temperature input
environment_tmp = st.sidebar.number_input("Environment Temperature:", min_value=-50.0, max_value=50.0, value=0.0)  # 环境温度输入框

# Power input
power = st.sidebar.number_input("Power:", min_value=-1000.0, max_value=1000.0, value=0.0)  # 功率输入框

# Wind speed input
wind_speed = st.sidebar.number_input("Wind Speed:", min_value=0.0, max_value=50.0, value=0.0)  # 风速输入框

# Pitch3 angle input
pitch3_angle = st.sidebar.number_input("Pitch 3 Angle:", min_value=-10.0, max_value=10.0, value=0.0)  # 桨叶3角度输入框

# Internal temperature input
int_tmp = st.sidebar.number_input("Internal Temperature:", min_value=-50.0, max_value=50.0, value=0.0)  # 内部温度输入框

# Pitch2 angle input
pitch2_angle = st.sidebar.number_input("Pitch 2 Angle:", min_value=-10.0, max_value=10.0, value=0.0)  # 桨叶2角度输入框

# Process the input and make a prediction
feature_values = [yaw_position, environment_tmp, power, wind_speed, pitch3_angle, int_tmp, pitch2_angle]  # 收集所有输入的特征
features = np.array([feature_values])  # 转换为NumPy数组

if st.button("Make Prediction"):  # 如果点击了预测按钮
    # Predict the class and probabilities
    predicted_class = model.predict(features)[0]  # 预测故障类别
    predicted_proba = model.predict_proba(features)[0]  # 预测各类别的概率

    # Display the prediction results
    st.write(f"**Predicted Class:** {predicted_class}")  # 显示预测的类别
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # 显示各类别的预测概率

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比

    if predicted_class == 1:  # 如果预测为故障
        advice = (
            f"According to our model, the wind turbine has a high risk of failure. "
            f"The probability of failure is {probability:.1f}%. "
            "Although this is just a probability estimate, it suggests that the turbine might be at a higher risk of failure. "
            "I recommend that you contact a maintenance team for further inspection and assessment, "
            "to ensure the turbine operates safely."
        )  # 如果预测为故障，给出相关建议
    else:  # 如果预测为正常
        advice = (
            f"According to our model, the wind turbine is operating normally. "
            f"The probability of normal operation is {probability:.1f}%. "
            "Nevertheless, regular maintenance and monitoring are still very important. "
            "I suggest that you continue to monitor the turbine's parameters regularly, "
            "and seek professional help if you notice any abnormal changes."
        )  # 如果预测为正常，给出相关建议

    st.write(advice)  # 显示建议

    # Visualize the prediction probabilities
    sample_prob = {
        'Class_0': predicted_proba[0],  # 类别0（正常）的概率
        'Class_1': predicted_proba[1]  # 类别1（故障）的概率
    }

    # Set figure size
    plt.figure(figsize=(10, 3))  # 设置图形大小

    # Create bar chart
    bars = plt.barh(['Normal', 'Failure'], 
                    [sample_prob['Class_0'], sample_prob['Class_1']], 
                    color=['#512b58', '#fe346e'])  # 绘制水平条形图

    # Add title and labels, set font bold and increase font size
    plt.title("Prediction Probability for Wind Turbine", fontsize=20, fontweight='bold')  # 添加图表标题，并设置字体大小和加粗
    plt.xlabel("Probability", fontsize=14, fontweight='bold')  # 添加X轴标签，并设置字体大小和加粗
    plt.ylabel("Classes", fontsize=14, fontweight='bold')  # 添加Y轴标签，并设置字体大小和加粗

    # Add probability text labels, adjust position to avoid overlap, set font bold
    for i, v in enumerate([sample_prob['Class_0'], sample_prob['Class_1']]):  # 为每个条形图添加概率文本标签
        plt.text(v + 0.0001, i, f"{v:.2f}", va='center', fontsize=14, color='black', fontweight='bold')  # 设置标签位置、字体加粗

    # Hide other axes (top, right, bottom)
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

    # Show the plot
    st.pyplot(plt)  # 显示图表