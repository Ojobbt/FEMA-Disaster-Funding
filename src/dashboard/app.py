from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_PATH = PROCESSED_DIR / "fema_merged.csv"


st.set_page_config(
    page_title="FEMA Dashboard",
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 0rem;
    }
    h1 {
        font-size: 1.7rem;
        margin-bottom: 0.1rem;
    }
    h2, h3 {
        font-size: 1rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "federalShareObligated" in df.columns:
        df["federalShareObligated"] = pd.to_numeric(
            df["federalShareObligated"], errors="coerce"
        ).fillna(0)

    if "dateObligated" in df.columns:
        df["dateObligated"] = pd.to_datetime(
            df["dateObligated"], errors="coerce"
        )

    return df


df = load_data(DATA_PATH)


st.title("FEMA Disaster Funding Dashboard")
st.caption("Public Assistance funding overview from FEMA OpenFEMA data")


required_cols = ["state_pa", "incidentType", "federalShareObligated"]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()


# Sidebar filters
st.sidebar.header("Filters")

states = sorted(df["state_pa"].dropna().unique())

selected_state = st.sidebar.selectbox(
    "State",
    states,
)

incident_types = sorted(df["incidentType"].dropna().unique())

selected_incidents = st.sidebar.multiselect(
    "Incident Type",
    incident_types,
)

filtered_df = df[df["state_pa"] == selected_state]

if selected_incidents:
    filtered_df = filtered_df[
        filtered_df["incidentType"].isin(selected_incidents)
    ]


if filtered_df.empty:
    st.warning("No data found for the selected filters.")
    st.write("Try selecting another state or clearing incident type filters.")
    st.stop()


# KPI row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "Total Funding",
    f"${filtered_df['federalShareObligated'].sum():,.0f}",
)

kpi2.metric(
    "Projects",
    f"{len(filtered_df):,}",
)

kpi3.metric(
    "Disasters",
    f"{filtered_df['disasterNumber'].nunique():,}"
    if "disasterNumber" in filtered_df.columns
    else "N/A",
)

kpi4.metric(
    "Counties",
    f"{filtered_df['county'].nunique():,}"
    if "county" in filtered_df.columns
    else "N/A",
)


# Charts
left, right = st.columns(2)

with left:
    st.subheader("Funding by Incident Type")

    incident_summary = (
        filtered_df.groupby("incidentType")["federalShareObligated"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(incident_summary, height=230)


with right:
    st.subheader("Top 10 Counties by Funding")

    if "county" in filtered_df.columns:
        county_summary = (
            filtered_df.groupby("county")["federalShareObligated"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(county_summary, height=230)
    else:
        st.info("County column not available.")


# Data table
st.subheader("Top FEMA Records")

display_cols = [
    "disasterNumber",
    "state_pa",
    "county",
    "applicantName",
    "incidentType",
    "federalShareObligated",
    "dateObligated",
]

available_cols = [col for col in display_cols if col in filtered_df.columns]

table_df = (
    filtered_df[available_cols]
    .sort_values("federalShareObligated", ascending=False)
    .head(15)
)

st.dataframe(
    table_df,
    use_container_width=True,
    height=210,
)


# Small debug footer
with st.expander("Data quality check"):
    st.write("Total rows:", len(df))
    st.write("Filtered rows:", len(filtered_df))
    st.write("Columns:", df.columns.tolist())