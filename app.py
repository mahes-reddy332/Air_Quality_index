import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Title ---
st.title("🌍 Air Quality Index (AQI) Dashboard")

# --- Load data from CSV ---
df = pd.read_csv("data_date.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.columns = df.columns.str.strip()

# --- Show raw data ---
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# --- Average AQI by Country ---
st.subheader("🌐 Average AQI by Country")
df_avg = df.groupby("Country")["AQI Value"].mean().reset_index(name="Average AQI").sort_values(by="Average AQI", ascending=False)

fig1, ax1 = plt.subplots()
sns.barplot(data=df_avg, x="Average AQI", y="Country", ax=ax1)
st.pyplot(fig1)
st.markdown("> This bar chart shows the **average Air Quality Index (AQI)** for each country. A higher AQI means more pollution. This helps compare overall air quality by country.")

# --- AQI Distribution by City ---
st.subheader("🏙️ AQI Distribution by City")
selected_country = st.selectbox("Select a Country", df["Country"].unique())
df_city = df[df["Country"] == selected_country]

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_city, x="City", y="AQI Value", ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)
st.markdown(f"> This boxplot shows **AQI distribution across cities** in {selected_country}. You can spot cities with high pollution or wide AQI variability.")

# --- AQI Trend Over Time ---
st.subheader("📈 AQI Trend Over Time")
selected_city = st.selectbox("Select a City", df["City"].unique())
df_time = df[df["City"] == selected_city]

fig3, ax3 = plt.subplots()
df_time.sort_values(by="Date", inplace=True)
sns.lineplot(data=df_time, x="Date", y="AQI Value", ax=ax3)
st.pyplot(fig3)
st.markdown(f"> This line chart tracks **AQI changes over time** in {selected_city}. It reveals seasonal or policy-driven variations.")

# --- AQI Category Count ---
st.subheader("📊 AQI Category Distribution")
fig4, ax4 = plt.subplots()
sns.countplot(data=df, x="AQI Category", order=df["AQI Category"].value_counts().index, ax=ax4)
plt.xticks(rotation=45)
st.pyplot(fig4)
st.markdown("> This chart shows the **frequency of different AQI categories** (e.g., Good, Moderate, Unhealthy). It gives a sense of how often air quality falls into each range.")

# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Seaborn")
