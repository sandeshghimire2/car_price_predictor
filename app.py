"""
Car Price Predictor — Streamlit App
------------------------------------
A web UI around the Linear Regression model trained in `maincode.ipynb`.

Run with:
    streamlit run app.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# Paths — anchored to this file's own folder, NOT the process's working
# directory. Streamlit Cloud (and some other launchers) don't guarantee the
# working directory is the app's folder, which causes FileNotFoundError on
# relative paths like "data/cleaned_car.csv".
# --------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "Linearmodelcar.pkl")
DATA_PATH = os.path.join(APP_DIR, "data", "cleaned_car.csv")

# --------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #ff4b4b, #ff9d4b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0rem;
        }
        .subtitle {
            color: #9a9a9a;
            font-size: 1.05rem;
            margin-top: 0rem;
            margin-bottom: 1.5rem;
        }
        .price-box {
            background: linear-gradient(135deg, #1f2937, #111827);
            border: 1px solid #ff4b4b33;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
        }
        .price-value {
            font-size: 3rem;
            font-weight: 800;
            color: #ff4b4b;
        }
        .price-label {
            color: #9a9a9a;
            font-size: 1rem;
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found at `{MODEL_PATH}`. Make sure "
            "`Linearmodelcar.pkl` was committed to your repository "
            "(check it isn't excluded by .gitignore or too large for a "
            "plain git push)."
        )
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(
            f"Data file not found at `{DATA_PATH}`. Make sure the `data/` "
            "folder (with `cleaned_car.csv` inside) was committed to your "
            "repository — it's easy to accidentally leave a data folder "
            "out of git."
        )
        st.stop()
    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


model = load_model()
df = load_data()

companies = sorted(df["company"].unique())
fuel_types = sorted(df["fuel_type"].unique())
year_min, year_max = int(df["year"].min()), int(df["year"].max())

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<p class="main-title">🚗 Car Price Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Estimate the resale price of a used car with a '
    "Linear Regression model trained on real listings.</p>",
    unsafe_allow_html=True,
)

tab_predict, tab_explore, tab_about = st.tabs(
    ["🔮 Predict Price", "📊 Explore Data", "ℹ️ About"]
)

# --------------------------------------------------------------------------
# TAB 1 — Prediction
# --------------------------------------------------------------------------
with tab_predict:
    col_form, col_result = st.columns([1.3, 1], gap="large")

    with col_form:
        st.subheader("Car details")

        company = st.selectbox("Company / Brand", companies, index=0)

        names_for_company = sorted(df[df["company"] == company]["name"].unique())
        name = st.selectbox("Model", names_for_company, index=0)

        col_a, col_b = st.columns(2)
        with col_a:
            year = st.slider(
                "Year of purchase",
                min_value=year_min,
                max_value=year_max,
                value=min(2015, year_max),
            )
        with col_b:
            fuel_type = st.selectbox("Fuel type", fuel_types)

        kms_driven = st.number_input(
            "Kilometers driven",
            min_value=0,
            max_value=500000,
            value=30000,
            step=1000,
        )

        predict_clicked = st.button("Predict Price 💰", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Estimated price")
        if predict_clicked:
            input_df = pd.DataFrame(
                [[name, company, year, kms_driven, fuel_type]],
                columns=["name", "company", "year", "kms_driven", "fuel_type"],
            )
            try:
                prediction = model.predict(input_df)[0]
                prediction = max(0, prediction)
                st.markdown(
                    f"""
                    <div class="price-box">
                        <div class="price-label">Predicted Price</div>
                        <div class="price-value">₹ {prediction:,.0f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    "This is an estimate based on historical listings and may "
                    "differ from the actual market price."
                )
            except Exception as e:
                st.error(f"Couldn't generate a prediction: {e}")
        else:
            st.info("Fill in the car details on the left and click **Predict Price** to see an estimate.")

# --------------------------------------------------------------------------
# TAB 2 — Data exploration
# --------------------------------------------------------------------------
with tab_explore:
    st.subheader("Dataset overview")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total listings", f"{len(df):,}")
    m2.metric("Brands", df["company"].nunique())
    m3.metric("Avg price", f"₹ {df['Price'].mean():,.0f}")
    m4.metric("Year range", f"{year_min}–{year_max}")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        avg_price_company = (
            df.groupby("company")["Price"].mean().sort_values(ascending=False).reset_index()
        )
        fig1 = px.bar(
            avg_price_company,
            x="company",
            y="Price",
            title="Average price by company",
            color="Price",
            color_continuous_scale="Reds",
        )
        fig1.update_layout(xaxis_title="", yaxis_title="Average price (₹)")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.scatter(
            df,
            x="kms_driven",
            y="Price",
            color="fuel_type",
            title="Price vs. kilometers driven",
            opacity=0.7,
        )
        fig2.update_layout(xaxis_title="Kilometers driven", yaxis_title="Price (₹)")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig3 = px.box(
            df,
            x="year",
            y="Price",
            title="Price distribution by year",
        )
        fig3.update_layout(xaxis_title="Year", yaxis_title="Price (₹)")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fuel_counts = df["fuel_type"].value_counts().reset_index()
        fuel_counts.columns = ["fuel_type", "count"]
        fig4 = px.pie(
            fuel_counts,
            names="fuel_type",
            values="count",
            title="Listings by fuel type",
            hole=0.45,
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("Browse the cleaned dataset")
    st.dataframe(df, use_container_width=True, height=350)

# --------------------------------------------------------------------------
# TAB 3 — About
# --------------------------------------------------------------------------
with tab_about:
    st.subheader("About this app")
    st.markdown(
        """
This app wraps a **Linear Regression** model (scikit-learn `Pipeline` with a
`OneHotEncoder` + `ColumnTransformer`) that predicts the resale price of a
used car based on:

- **Name** (brand + model)
- **Company**
- **Year of purchase**
- **Kilometers driven**
- **Fuel type**

The model was trained on cleaned data derived from `car.csv` — see
`train_model.py` for the full data-cleaning and training pipeline (ported
directly from the original `maincode.ipynb` notebook).

**Project structure**
