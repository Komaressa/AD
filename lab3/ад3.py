import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Vegetation Index Viewer", layout="wide")

# Load dataset with caching
@st.cache_data
def get_dataset():
    return pd.read_csv("full.csv")

# Load data
data = get_dataset()

# Sidebar UI
with st.sidebar:
    st.title("Filter Options")

    index_type = st.radio("Index Type", options=["VCI", "TCI", "VHI"])
    province = st.selectbox("Province ID", sorted(data["PROVINCE_ID"].unique()))

    week_range = st.slider(
        "Select Week Range",
        min_value=int(data["Week"].min()),
        max_value=int(data["Week"].max()),
        value=(int(data["Week"].min()), int(data["Week"].max()))
    )

    year_range = st.slider(
        "Select Year Range",
        min_value=int(data["Year"].min()),
        max_value=int(data["Year"].max()),
        value=(int(data["Year"].min()), int(data["Year"].max()))
    )

    ascending_sort = st.checkbox(f"Sort {index_type} Ascending")
    descending_sort = st.checkbox(f"Sort {index_type} Descending")

# Filter data
subset = data[
    (data["PROVINCE_ID"] == province) &
    (data["Week"].between(*week_range)) &
    (data["Year"].between(*year_range))
]

# Handle sorting logic
if ascending_sort and descending_sort:
    st.warning("Both sort orders selected. Showing ascending by default.")
    subset = subset.sort_values(by=index_type)
elif ascending_sort:
    subset = subset.sort_values(by=index_type)
elif descending_sort:
    subset = subset.sort_values(by=index_type, ascending=False)

# Tabs for visualizations
view_tab, line_tab, compare_tab = st.tabs(["Table View", "Line Chart", "Province Compare"])

# Table View Tab
with view_tab:
    st.subheader("Filtered Dataset")
    st.dataframe(subset)

# Line Chart Tab
with line_tab:
    st.subheader(f"{index_type} Time Series for Province {province}")
    time = subset["Year"] + (subset["Week"] - 1) / 52
    fig_line = px.line(
        subset,
        x=time,
        y=index_type,
        title=f"{index_type} over Time",
        labels={"x": "Year.Week", index_type: f"{index_type} Value"},
        template="plotly_dark"
    )
    st.plotly_chart(fig_line, use_container_width=True)

# Comparison Tab
with compare_tab:
    st.subheader(f"{index_type} Comparison Across Provinces")
    comp_data = data[
        (data["Week"].between(*week_range)) &
        (data["Year"].between(*year_range))
    ]

    fig_compare = go.Figure()

    # Highlight selected province
    target = comp_data[comp_data["PROVINCE_ID"] == province]
    fig_compare.add_trace(
        go.Scatter(
            x=target["Year"] + (target["Week"] - 1) / 52,
            y=target[index_type],
            mode="lines",
            name=f"Province {province}",
            line=dict(color="red", width=3)
        )
    )

    # Plot other provinces
    for pid in sorted(data["PROVINCE_ID"].unique()):
        if pid != province:
            temp = comp_data[comp_data["PROVINCE_ID"] == pid]
            fig_compare.add_trace(
                go.Scatter(
                    x=temp["Year"] + (temp["Week"] - 1) / 52,
                    y=temp[index_type],
                    mode="lines",
                    name=f"Province {pid}",
                    line=dict(color="gray", width=1),
                    opacity=0.2
                )
            )

    fig_compare.update_layout(
        title=f"{index_type} Index Across Provinces",
        xaxis_title="Year.Week",
        yaxis_title=f"{index_type} Value",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_compare, use_container_width=True)

# Optional custom style
st.markdown("""
<style>
    .stApp {
        background-color: #f9fafb;
    }
    .stSidebar {
        background-color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)
