import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import expit
import re

# ==================== 特征定义（连续变量）====================
# key: 模型中的变量名, display: 页面展示名称
FEATURE_CONFIG = {
    'zj':       {'display': 'Tumor size'},
    'CYFRA21-1': {'display': 'Cytokeratin 19 fragment 21-1 (CYFRA21-1)'},
    'CA125':    {'display': 'Cancer antigen 125 (CA125)'},
    'NLR':      {'display': 'Neutrophil to Lymphocyte Ratio(NLR)'},
    'CTR':      {'display': 'Consolidation to Tumor Ratio(CTR)'},
    'LMR':      {'display': 'Lymphocyte to Monocyte Ratio(LMR)'},
    'PT':       {'display': 'Prothrombin Time(PT)'},
    'PLR':      {'display': 'Platelet to Lymphocyte Ratio(PLR)'},
    'MO':       {'display': 'Monocytes(Mo)'},
    'NSE':      {'display': 'Neuron-Specific Enolase (NSE)'}
}

# ==================== 模型加载 ====================
@st.cache_resource
def load_model():
    model_path = Path("best_catboost_model.pkl")
    #model_path = Path(r"E:\lc-chengxu\ling\best_catboost_model.pkl") # 本地试验时: 
    model = joblib.load(model_path)
    
    try:
        explainer = shap.TreeExplainer(model, model_output="probability")
    except Exception:
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            background = pd.DataFrame(np.zeros((1, 10)), columns=list(FEATURE_CONFIG.keys()))
            explainer = shap.KernelExplainer(model.predict_proba, background)
    return model, explainer

# ==================== 输入收集（连续变量数值输入）====================
def collect_input():
    input_dict = {}
    for feat, config in FEATURE_CONFIG.items():
        st.markdown(f"**{config['display']}**")
        val = st.number_input(
            "",
            value=0.0,
            format="%.2f",  # ← 改为 2 位小数
            key=feat,
            label_visibility="collapsed"
        )
        input_dict[feat] = val
    return pd.DataFrame([input_dict])

# ==================== 力图（概率刻度 + 纯净标签）====================
def plot_force_only(explainer, input_df, model, prob):
    if isinstance(explainer, shap.KernelExplainer):
        shap_values = explainer.shap_values(input_df)
        sv = shap_values[1][0]
        base = explainer.expected_value[1]
    else:
        result = explainer(input_df)
        if isinstance(result, shap.Explanation):
            if len(result.values.shape) == 3:
                sv = result.values[0, :, 1]
                base = result.base_values[0, 1] if len(result.base_values.shape) > 1 else result.base_values[1]
            else:
                sv = result.values[0]
                base = result.base_values[0]
        else:
            sv = result[1][0]
            base = explainer.expected_value[1] if hasattr(explainer, 'expected_value') else 0.5

    # 构建特征显示名称：展示名 + 实际数值
    feature_names = []
    for feat in input_df.columns:
        val = input_df[feat].iloc[0]
        display_name = FEATURE_CONFIG[feat]['display']
        feature_names.append(f"{display_name}: {val:.2f}")  # ← 改为 2 位小数

    plt.figure(figsize=(24, 12), dpi=120)
    shap.force_plot(
        base_value=float(base),
        shap_values=np.array(sv),
        features=input_df.iloc[0].values,
        feature_names=feature_names,
        link="identity",
        matplotlib=True,
        show=False
    )

    ax = plt.gca()
    final_margin = float(base) + float(np.sum(sv))

    # x轴刻度改为概率值
    ticks = ax.get_xticks()
    prob_labels = []
    for t in ticks:
        if -25 < t < 25:
            prob_labels.append(f"{expit(t):.2f}")  # ← 改为 2 位小数
        else:
            prob_labels.append("")
    ax.set_xticklabels(prob_labels, fontsize=10)
    ax.set_xlabel("")

    # 处理标签：f(x)改为概率，去掉数值后缀中的重复值
    for text in ax.texts:
        txt = text.get_text()
        # 将 f(x) 标签改为该样本的具体概率值
        if re.search(r'f\(x\)', txt):
            text.set_text(f"f(x) ")
        # 处理可能的独立数值标签
        elif re.match(r'^[-+]?\d+\.?\d*$', txt.strip()):
            try:
                val = float(txt.strip())
                if abs(val - final_margin) < 0.05:
                    text.set_text(f"{prob:.2f}")  # ← 改为 2 位小数
                    x, y = text.get_position()
                    text.set_y(y - 0.04)
            except ValueError:
                pass
        # 去掉特征标签末尾的 "= 数值" 后缀（shap自动添加的）
        # 因为我们在 feature_names 中已经手动加了数值，这里需要去掉shap重复添加的部分
        elif re.search(r'=\s*[-+]?\d+\.?\d*\s*$', txt):
            clean_txt = txt.split('=')[0].strip()
            text.set_text(clean_txt)

    plt.tight_layout()
    fig = plt.gcf()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ==================== 主程序 ====================
def main():
    st.title("A Predictive Model for LNM in Clinical T1 Lung Cancer with CTR > 0.5")
    model, explainer = load_model()
    input_df = collect_input()

    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ffffff;
        color: #ff4b4b;
        border: 2px solid #ff4b4b;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 120px;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff4b4b;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Predict", key="predict_btn"):
        with st.spinner("Computing..."):
            prob = model.predict_proba(input_df)[0, 1]
            st.markdown(f"*Based on feature values, predicted possibility of LNM is {prob:.1%}*")
            plot_force_only(explainer, input_df, model, prob)

if __name__ == "__main__":
    main()
