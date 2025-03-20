# app.py
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dowhy import CausalModel

# 生成模擬腎臟病學數據
def generate_data(n=1000, dm_rate=0.3, htn_rate=0.4, acei_effect=2):
    np.random.seed(42)
    age = np.random.normal(60, 10, n)
    sex = np.random.choice([0, 1], size=n)
    dm = np.random.choice([0, 1], size=n, p=[1-dm_rate, dm_rate])
    htn = np.random.choice([0, 1], size=n, p=[1-htn_rate, htn_rate])
    baseline_egfr = np.random.normal(75, 15, n)
    acei_arb = ((baseline_egfr < 90) & (htn == 1)).astype(int)
    uacr = np.random.normal(50, 20, n) + 20 * dm + 10 * htn - 15 * acei_arb
    potassium = np.random.normal(4.5, 0.5, n) + 0.2 * acei_arb - 0.1 * (baseline_egfr / 10)
    final_egfr = baseline_egfr - (0.3 * age) - (5 * dm) - (3 * htn) + (acei_effect * acei_arb) - (0.2 * uacr)

    return pd.DataFrame({
        "age": age, "sex": sex, "dm": dm, "htn": htn, "baseline_egfr": baseline_egfr,
        "acei_arb": acei_arb, "uacr": uacr, "potassium": potassium, "final_egfr": final_egfr
    })

# DAG 繪圖函數
def plot_dag():
    G = nx.DiGraph()
    nodes = ["ACEI/ARB", "Final eGFR", "Baseline eGFR", "Age", "Sex", "DM", "Hypertension", "UACR（中介變數）", "Potassium（對撞因子）"]
    edges = [
        ("ACEI/ARB", "Final eGFR"),  
        ("Baseline eGFR", "Final eGFR"),
        ("Baseline eGFR", "ACEI/ARB"),  
        ("DM", "Final eGFR"), ("Hypertension", "Final eGFR"),
        ("DM", "ACEI/ARB"), ("Hypertension", "ACEI/ARB"),
        ("UACR（中介變數）", "Final eGFR"), ("DM", "UACR（中介變數）"), ("Hypertension", "UACR（中介變數）"),
        ("ACEI/ARB", "UACR（中介變數）"),
        ("ACEI/ARB", "Potassium（對撞因子）"), ("Final eGFR", "Potassium（對撞因子）")
    ]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=3000, font_size=10)
    plt.title("ACEI/ARB 與 eGFR 的因果 DAG")
    st.pyplot(plt)

# Streamlit UI
st.title("腎臟病因果推斷 Streamlit 應用")

n_samples = st.sidebar.number_input("樣本數", 100, 5000, 1000, step=100)
dm_rate = st.sidebar.slider("糖尿病比例", 0.1, 0.5, 0.3)
htn_rate = st.sidebar.slider("高血壓比例", 0.1, 0.5, 0.4)
acei_effect = st.sidebar.slider("ACEI/ARB 對 eGFR 影響", 0, 5, 2)
if st.sidebar.button("生成數據"):
    data = generate_data(n=n_samples, dm_rate=dm_rate, htn_rate=htn_rate, acei_effect=acei_effect)
    st.write("數據預覽：", data.head())

if st.sidebar.button("顯示 DAG"):
    plot_dag()

adjust_set = st.multiselect("選擇調整變數", 
                            ["baseline_egfr", "age", "sex", "dm", "htn", "uacr", "potassium"])

if st.button("估計因果效應"):
    if "uacr" in adjust_set:
        st.warning("⚠️ UACR 是中介變數，不應納入調整！")
    if "potassium" in adjust_set:
        st.warning("⚠️ 血鉀（Potassium）是對撞因子，不應納入調整！")
    
    try:
        data = generate_data(n=n_samples, dm_rate=dm_rate, htn_rate=htn_rate, acei_effect=acei_effect)
        model = CausalModel(
            data=data,
            treatment="acei_arb",
            outcome="final_egfr",
            common_causes=adjust_set
        )
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
        st.success(f"✅ ACEI/ARB 對 eGFR 的估計因果效應：{estimate.value}")
    except Exception as e:
        st.error(f"❌ 錯誤：{str(e)}")
