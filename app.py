import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Retail Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================
# MODERN BLUE UI
# =========================
st.markdown("""
<style>

/* Main App Background */
.stApp {

    background:
    linear-gradient(
        135deg,
        #0F172A 0%,
        #1E3A8A 40%,
        #2563EB 100%
    );

    color: white;
}

/* Hide Streamlit Branding */
#MainMenu {
    visibility:hidden;
}

footer {
    visibility:hidden;
}

header {
    visibility:hidden;
}

/* Global Text */
html, body, [class*="css"] {
    color: white;
}

/* Main Container */
.block-container {
    padding-top: 2rem;
}

/* KPI Cards */
div[data-testid="metric-container"] {

    background:
    rgba(255,255,255,0.08);

    backdrop-filter: blur(12px);

    border:
    1px solid rgba(255,255,255,0.15);

    padding: 20px;

    border-radius: 20px;

    box-shadow:
    0px 8px 24px rgba(0,0,0,0.25);
}

/* KPI Labels */
div[data-testid="metric-container"] label {
    color: #E2E8F0 !important;
}

/* KPI Values */
div[data-testid="metric-container"] div {
    color: white !important;
}

/* Tabs */
button[data-baseweb="tab"] {

    background:
    rgba(255,255,255,0.08);

    color: white !important;

    border-radius: 12px;

    margin-right: 8px;

    padding: 10px 18px;

    font-size: 15px;

    font-weight: 600;
}

/* Selected Tab */
button[aria-selected="true"] {

    background:
    linear-gradient(
        135deg,
        #3B82F6,
        #2563EB
    ) !important;

    color: white !important;
}

/* Buttons */
.stButton > button {

    background:
    linear-gradient(
        135deg,
        #60A5FA,
        #2563EB
    );

    color: white;

    border: none;

    border-radius: 14px;

    padding: 12px 20px;

    font-weight: 600;

    transition: 0.3s;
}

/* Button Hover */
.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
    0px 6px 18px rgba(37,99,235,0.4);
}

/* Download Button */
.stDownloadButton > button {

    background:
    linear-gradient(
        135deg,
        #60A5FA,
        #2563EB
    );

    color: white;

    border: none;

    border-radius: 14px;

    padding: 12px 20px;

    font-weight: 600;
}

/* Dataframes */
[data-testid="stDataFrame"] {

    border-radius: 16px;

    overflow:hidden;

    border:
    1px solid rgba(255,255,255,0.15);
}

/* Input Fields */
.stNumberInput input {

    background:
    rgba(255,255,255,0.08);

    color: white !important;

    border-radius: 10px;
}

/* Plot Containers */
.element-container {

    background:
    rgba(255,255,255,0.04);

    padding: 10px;

    border-radius: 16px;
}

/* Markdown Text */
p, li {
    color: #E2E8F0;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(
    "Sample - Superstore.csv",
    encoding="latin1"
)

# =========================
# CLEAN COLUMNS
# =========================
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.lower()
)

# =========================
# DATE CONVERSION
# =========================
df["order_date"] = pd.to_datetime(
    df["order_date"],
    errors="coerce"
)

df = df.dropna(subset=["order_date"])

# =========================
# HEADER
# =========================
st.title("Retail Sales Analytics Dashboard")

st.markdown("""
## Business Intelligence Dashboard

Analyze:
- Sales Performance
- Regional Profit
- Customer Trends
- Predictive Analytics
- Forecasting Analysis
""")

# =========================
# KPI SECTION
# =========================
total_sales = df["sales"].sum()
total_profit = df["profit"].sum()
total_orders = df["order_id"].nunique()
total_customers = df["customer_id"].nunique()

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total Sales",
    f"${total_sales:,.0f}"
)

k2.metric(
    "Total Profit",
    f"${total_profit:,.0f}"
)

k3.metric(
    "Orders",
    total_orders
)

k4.metric(
    "Customers",
    total_customers
)

st.markdown("---")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "EDA Analysis",
    "Machine Learning",
    "Business Insights"
])

# =========================
# OVERVIEW TAB
# =========================
with tab1:

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Sales by Category")

        category_sales = (
            df.groupby("category")["sales"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            category_sales,
            x="sales",
            y="category",
            orientation="h",
            color="sales"
        )

        fig.update_layout(
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        st.subheader("Profit by Region")

        region_profit = (
            df.groupby("region")["profit"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            region_profit,
            x="region",
            y="profit",
            color="profit"
        )

        fig.update_layout(
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# =========================
# EDA ANALYSIS TAB
# =========================
with tab2:

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Monthly Sales Trend")

        monthly_sales = (
            df.groupby(
                df["order_date"].dt.to_period("M")
            )["sales"]
            .sum()
            .reset_index()
        )

        monthly_sales["order_date"] = (
            monthly_sales["order_date"]
            .astype(str)
        )

        fig = px.line(
            monthly_sales,
            x="order_date",
            y="sales",
            markers=True
        )

        fig.update_layout(
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        st.subheader("Sales Distribution")

        fig = px.histogram(
            df,
            x="sales",
            nbins=30
        )

        fig.update_layout(
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    fig, ax = plt.subplots(figsize=(4,2))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# =========================
# MACHINE LEARNING TAB
# =========================
with tab3:

    st.header("Sales Prediction & Forecasting System")

    ml_df = df[[
        "sales",
        "profit",
        "quantity",
        "discount"
    ]]

    X = ml_df[[
        "quantity",
        "discount",
        "profit"
    ]]

    y = ml_df["sales"]

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =========================
    # LINEAR REGRESSION
    # =========================
    lr_model = LinearRegression()

    lr_model.fit(X_train, y_train)

    lr_predictions = lr_model.predict(X_test)

    lr_r2 = r2_score(
        y_test,
        lr_predictions
    )

    lr_mae = mean_absolute_error(
        y_test,
        lr_predictions
    )

    # =========================
    # RANDOM FOREST
    # =========================
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)

    rf_r2 = r2_score(
        y_test,
        rf_predictions
    )

    rf_mae = mean_absolute_error(
        y_test,
        rf_predictions
    )

    # =========================
    # MODEL COMPARISON
    # =========================
    st.subheader("Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest"
        ],
        "R2 Score": [
            lr_r2,
            rf_r2
        ],
        "MAE": [
            lr_mae,
            rf_mae
        ]
    })

    st.dataframe(comparison_df)

    fig = px.bar(
        comparison_df,
        x="Model",
        y="R2 Score",
        color="Model",
        title="Model Accuracy Comparison"
    )

    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # RANDOM FOREST PREDICTIONS
    # =========================
    st.subheader("Random Forest Predictions")

    fig = px.scatter(
        x=y_test,
        y=rf_predictions,
        labels={
            "x":"Actual Sales",
            "y":"Predicted Sales"
        }
    )

    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # FORECASTING
    # =========================
    st.subheader("Sales Forecasting")

    forecast_df = (
        df.groupby(
            df["order_date"].dt.to_period("M")
        )["sales"]
        .sum()
        .reset_index()
    )

    forecast_df["order_date"] = (
        forecast_df["order_date"]
        .astype(str)
    )

    forecast_df["Forecast"] = (
        forecast_df["sales"]
        .rolling(3)
        .mean()
    )

    fig = px.line(
        forecast_df,
        x="order_date",
        y=[
            "sales",
            "Forecast"
        ],
        markers=True,
        title="Sales Forecast Trend"
    )

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # USER PREDICTION
    # =========================
    st.subheader("Predict Future Sales")

    c1, c2, c3 = st.columns(3)

    with c1:

        quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=2
        )

    with c2:

        discount = st.number_input(
            "Discount",
            min_value=0.0,
            max_value=1.0,
            value=0.1
        )

    with c3:

        profit = st.number_input(
            "Profit",
            value=50.0
        )

    if st.button("Predict Sales"):

        input_data = np.array([
            [quantity, discount, profit]
        ])

        prediction = rf_model.predict(
            input_data
        )

        st.success(
            f"Predicted Sales: ${prediction[0]:.2f}"
        )

    # =========================
    # DOWNLOAD REPORT
    # =========================
    results_df = pd.DataFrame({
        "Actual Sales": y_test.values,
        "Predicted Sales": rf_predictions
    })

    csv = results_df.to_csv(index=False)

    st.download_button(
        label="Download Prediction Report",
        data=csv,
        file_name="sales_prediction_report.csv",
        mime="text/csv"
    )

# =========================
# BUSINESS INSIGHTS TAB
# =========================
with tab4:

    st.header("Business Insights & Conclusions")

    st.markdown("""

### Key Findings

- Technology category generated the highest revenue.
- Some regions contribute significantly higher profits.
- Discounts negatively affect profit margins.
- Monthly sales fluctuate based on seasonal demand.
- Random Forest produced better prediction accuracy than Linear Regression.
- Forecasting helps businesses estimate future sales trends.

### Conclusion

This project demonstrates how businesses can use:
- Data Analytics
- Visualization
- Machine Learning
- Forecasting

to improve business decisions and optimize sales strategy.

""")

    st.subheader("Top Profitable Categories")

    profit_category = (
        df.groupby("category")["profit"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        profit_category,
        names="category",
        values="profit"
    )

    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown("""
Developed by Krithik

Retail Sales Analytics Dashboard
""")