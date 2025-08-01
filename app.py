import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sqlalchemy import create_engine
import urllib.parse

st.set_page_config(layout="wide")
st.title("🌍 Air Quality Index (AQI) Global Analysis")

# --- Database connection ---
username = 'root'
password = urllib.parse.quote_plus('Mahes@123Zz')  # Replace securely!
host = '127.0.0.1'
port = 3306
database = 'aqi_project'
conn_str = f'mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}'
engine = create_engine(conn_str)

# --- Load data ---
df = pd.read_sql("SELECT * FROM urs", engine)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.columns = df.columns.str.strip()

# --- Helper ---
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8]:
        return 'Monsoon'
    else:
        return 'Autumn'

df['Season'] = df['Date'].dt.month.apply(get_season)
df['month'] = df['Date'].dt.to_period('M').astype(str)

# --- SQL Results ---
st.header("🔍 SQL Insights")
col1, col2 = st.columns(2)

with col1:
    df_avg = pd.read_sql("""
        SELECT country, AVG(aqi_value) AS avg_aqi
        FROM air_quality
        GROUP BY country
        ORDER BY avg_aqi DESC
    """, engine)
    st.subheader("Top 10 Polluted Countries")
    sns.barplot(data=df_avg.head(10), x="avg_aqi", y="country")
    st.pyplot(plt.gcf())
    plt.clf()

with col2:
    df_status = pd.read_sql("SELECT status, COUNT(*) AS count FROM air_quality GROUP BY status", engine)
    df_status.set_index("status").plot.pie(y="count", autopct="%.1f%%", legend=False)
    plt.title("AQI Status Distribution")
    plt.ylabel("")
    st.pyplot(plt.gcf())
    plt.clf()

# --- Trend Chart ---
df_trend = pd.read_sql("SELECT data_date, AVG(aqi_value) AS daily_avg FROM air_quality GROUP BY data_date ORDER BY data_date", engine)
st.subheader("📈 AQI Trend Over Time")
fig_trend = px.line(df_trend, x="data_date", y="daily_avg", title="Global AQI Trend Over Time")
st.plotly_chart(fig_trend, use_container_width=True)

# --- Choropleth Map ---
st.header("🗺️ Global AQI Map")
fig_map = px.choropleth(df_avg, locations="country", locationmode="country names",
                        color="avg_aqi", range_color=(0, 500),
                        color_continuous_scale="RdYlGn_r",
                        labels={'avg_aqi': 'Average AQI'},
                        title="Average AQI by Country")
fig_map.update_geos(showcoastlines=True, projection_type="natural earth")
st.plotly_chart(fig_map, use_container_width=True)

# --- Seasonal Maps ---
df_grouped = df.groupby(['Country', 'Season'])['AQI Value'].mean().reset_index()
df_grouped.rename(columns={'AQI Value': 'avg_aqi'}, inplace=True)
st.subheader("📅 Seasonal AQI Maps")
season = st.selectbox("Select Season", ['Winter', 'Summer', 'Monsoon', 'Autumn'])
df_season = df_grouped[df_grouped['Season'] == season]
fig_season = px.choropleth(df_season, locations="Country", locationmode="country names",
                           color="avg_aqi", range_color=(0, 500),
                           color_continuous_scale="RdYlGn_r",
                           title=f"AQI Map - {season}")
st.plotly_chart(fig_season, use_container_width=True)

# --- Violation Summary ---
st.header("❌ AQI Violations by Country")
df['is_violation'] = df['AQI Value'] > 100
violation_summary = df.groupby('Country')['is_violation'].mean().sort_values(ascending=False)
top_n = 40
top_violators = violation_summary.head(top_n)
others = violation_summary.iloc[top_n:].sum()
violation_summary_grouped = top_violators.copy()
violation_summary_grouped['Others'] = others

plt.figure(figsize=(12, 6))
violation_summary_grouped.plot(kind='bar', color='tomato')
plt.title("Top AQI Violating Countries (Others Grouped)")
plt.ylabel("Violation Rate")
plt.xlabel("Country")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

# --- Country-wise Monthly Trend ---
st.header("📊 Country-Specific Monthly AQI Trend")
countries = sorted(df['Country'].dropna().unique())
selected_country = st.selectbox("Select a Country", countries)
df_country = df[df['Country'] == selected_country]

if not df_country.empty:
    df_monthly = df_country.groupby('month')['AQI Value'].mean().reset_index()
    fig_month = px.line(df_monthly, x='month', y='AQI Value',
                        title=f"Monthly AQI Trend - {selected_country}",
                        markers=True)
    st.plotly_chart(fig_month, use_container_width=True)
    avg_aqi = df_country['AQI Value'].mean()
    best = df_monthly.loc[df_monthly['AQI Value'].idxmin()]
    worst = df_monthly.loc[df_monthly['AQI Value'].idxmax()]
    st.info(f"✅ Average AQI: {avg_aqi:.2f}")
    st.success(f"🟢 Best Month: {best['month']} ({best['AQI Value']:.2f})")
    st.error(f"🔴 Worst Month: {worst['month']} ({worst['AQI Value']:.2f})")
else:
    st.warning("No data available for this country.")
