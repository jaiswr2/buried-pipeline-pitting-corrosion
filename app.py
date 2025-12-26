import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ================================
# Load trained model & feature space
# ================================
model = joblib.load("catboost_dmax_model.pkl")
feature_columns = joblib.load("synthetic_feature_columns.pkl")

# ================================
# Page setup
# ================================
st.set_page_config(
    page_title="Steel Pile Corrosion Predictor",
    layout="wide"
)

# ================================
# Global styling
# ================================
st.markdown(
    """
    <style>
    /* Main headings */
    h1, h2, h3 {
    	color: #7A003C !important;
    	font-weight: 700 !important;
     }


    /* Input labels */
    label,
    .stNumberInput label,
    .stSelectbox label {
        color: cornflowerblue !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }



    /* Subheaders like Input Parameters */
    .css-10trblm {
        color: cornflowerblue;
	font-weight: 700 !important;  /* bold */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================================
# Category labels
# ================================
COATING_LABELS = {
    "NC":  "Noncoated (NC)",
    "AEC": "Asphalt-Enamel Coated (AEC)",
    "WTC": "Wrap-Tape Coated (WTC)",
    "CTC": "Coal-Tar Coated (CTC)",
    "FBE": "Fusion-Bonded Epoxy (FBE)"
}

SOILCLASS_LABELS = {
    "C":   "Clay (C)",
    "CL":  "Clay Loam (CL)",
    "SCL": "Sandy Clay Loam (SCL)"
}

COATING_CODES = {v: k for k, v in COATING_LABELS.items()}
SOILCLASS_CODES = {v: k for k, v in SOILCLASS_LABELS.items()}

# ================================
# Typical (5–95%) ranges
# ================================
R = {
    "t (years)": (10.0, 50.0),
    "pH": (5.0, 8.0),
    "pp(A) (V)": (-1.5, -0.6),
    "re (Ω·m)": (5.0, 200.0),
    "wc (%)": (15.0, 40.0),
    "bd (g/mL)": (1.18, 1.42),
    "cc (ppm)": (10.0, 200.0),
    "bc (ppm)": (6.0, 80.0),
    "sc (ppm)": (20.0, 600.0),
    "rp (mV) (B)": (30.0, 300.0),
}

# ================================
# Header
# ================================


st.title("Prediction of Maximum Pitting Corrosion Depth (dₘₐₓ) in Buried Steel Pipeline")



st.divider()

# ================================
# Layout
# ================================
left, right = st.columns([1.3, 1.7], gap="large")

# ================================
# INPUTS (LEFT)
# ================================
with left:
    st.markdown(
    "<h3 style='color: cornflowerblue; font-weight:700; font-size:18px;'>Input Parameters</h3>",
    unsafe_allow_html=True
    )


    c1, c2 = st.columns(2)

    with c1:
        t_years = st.number_input(
            f"Exposure time, t (years)  [{R['t (years)'][0]}–{R['t (years)'][1]}]",
            1, 200, 25, step=1
        )
        pH = st.number_input(
            f"Soil pH  [{R['pH'][0]}–{R['pH'][1]}]",
            3.0, 10.0, 6.5, step=0.1
        )
        pp = st.number_input(
            f"Pipe-to-soil potential (V)  [{R['pp(A) (V)'][0]}–{R['pp(A) (V)'][1]}]",
            -2.5, 0.5, -0.8, step=0.05
        )
        re = st.number_input(
            f"Soil resistivity (Ω·m)  [{R['re (Ω·m)'][0]}–{R['re (Ω·m)'][1]}]",
            0.1, 2000.0, 30.0, step=1.0
        )
        wc = st.number_input(
            f"Water content (%)  [{R['wc (%)'][0]}–{R['wc (%)'][1]}]",
            0.0, 100.0, 25.0, step=1.0
        )
        bd = st.number_input(
            f"Bulk density (g/mL)  [{R['bd (g/mL)'][0]}–{R['bd (g/mL)'][1]}]",
            0.5, 2.5, 1.30, step=0.01
        )

    with c2:
        cc = st.number_input(
            f"Chloride (ppm)  [{R['cc (ppm)'][0]}–{R['cc (ppm)'][1]}]",
            0.0, 50000.0, 50.0, step=5.0
        )
        bc = st.number_input(
            f"Bicarbonate (ppm)  [{R['bc (ppm)'][0]}–{R['bc (ppm)'][1]}]",
            0.0, 50000.0, 20.0, step=5.0
        )
        sc = st.number_input(
            f"Sulfate (ppm)  [{R['sc (ppm)'][0]}–{R['sc (ppm)'][1]}]",
            0.0, 50000.0, 150.0, step=10.0
        )
        rp = st.number_input(
            f"Redox potential (mV)  [{R['rp (mV) (B)'][0]}–{R['rp (mV) (B)'][1]}]",
            -500.0, 800.0, 150.0, step=10.0
        )

        coating_label = st.selectbox("Pipeline coating type", list(COATING_CODES.keys()))
        soilclass_label = st.selectbox("Soil textural class", list(SOILCLASS_CODES.keys()))

    ct_code = COATING_CODES[coating_label]
    soil_code = SOILCLASS_CODES[soilclass_label]

    st.divider()

    if st.button("Predict", use_container_width=True):
        input_dict = {
            "t (years)": t_years,
            "pH": pH,
            "pp(A) (V)": pp,
            "re (Ω·m)": re,
            "wc (%)": wc,
            "bd (g/mL)": bd,
            "cc (ppm)": cc,
            "bc (ppm)": bc,
            "sc (ppm)": sc,
            "rp (mV) (B)": rp,
            "ct ( C )": ct_code,
            "Class (D)": soil_code
        }

        input_df = pd.DataFrame([input_dict])
        enc = pd.get_dummies(
            input_df,
            columns=["ct ( C )", "Class (D)"],
            drop_first=False
        ).reindex(columns=feature_columns, fill_value=0)

        st.session_state["base_df"] = input_df
        st.session_state["dmax"] = float(model.predict(enc)[0])

# ================================
# OUTPUT + PLOT (RIGHT)
# ================================
with right:

    if "dmax" in st.session_state:
        st.markdown(
            f"""
            <div style="color:#7A003C; font-weight:700; font-size:18px;">
            Predicted dₘₐₓ at t = {t_years} years<br>
            <span style="font-size:26px;">{st.session_state['dmax']:.2f} mm</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    if "base_df" in st.session_state:
        base = st.session_state["base_df"]
        t_pts = np.array([1, 10, 20, 30, 40])
        d_pts = []

        for t in t_pts:
            tmp = base.copy()
            tmp["t (years)"] = t
            tmp_enc = pd.get_dummies(
                tmp,
                columns=["ct ( C )", "Class (D)"],
                drop_first=False
            ).reindex(columns=feature_columns, fill_value=0)

            d_pts.append(model.predict(tmp_enc)[0])

        fig = plt.figure(figsize=(3.6, 2.4))
        plt.plot(t_pts, d_pts, color="black", marker="o", linewidth=1.6)
        plt.xlabel("Time (years)", fontsize=8)
        plt.ylabel("dₘₐₓ (mm)", fontsize=8)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout(pad=0.6)

        st.pyplot(fig, clear_figure=True, use_container_width=False)


