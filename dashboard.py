"""
streamlit run dashboard.py
"""

from openai import OpenAI
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.title("cudf GitHub issue explorer for DevRel")

st.markdown(
    """
This dashboard can help with:
- Identification of users/leads
- Identify common pain points
- Identify most common requested features
"""
)

st.markdown(
    "**This dashboard is WIP and may be prone to errors.** "
    "Code associated with this can be found at "
    "https://github.com/raybellwaves/cudf-issue-explorer"
)

st.markdown(
    "We can explore what companies are active in in the cudf GitHub repo. "
    "These are likely super users"
)


@st.cache_data
def fetch_community_data():
    return pd.read_parquet("all_poster_commenter_details.parquet")


@st.cache_data
def fetch_issue_data():
    return pd.read_parquet("issue_details_with_posters_small.parquet")


df_community = fetch_community_data()
df_issues = fetch_issue_data()

external_users = df_community[df_community["is_nvidia_employee"] == False]
company_counts = external_users["company"].value_counts()

# Plotting the histogram
fig, ax = plt.subplots(figsize=(10, 6))

company_counts.plot(
    kind="bar",
    ax=ax,
    xlabel="Company",
    ylabel="Count",
    title="Count of companies who post and comment on the cudf GitHub repo",
)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig)

st.markdown(
    "To learn more about these companies we will drop into a chat session. "
    "Here you can ask questions such as 'What type of company is Halliburton?'. "
    "Or generic questions such 'Why would Halliburton use GPUs to speed up data processing?' "
    "but don't expect a good result. "
    "We will use the GitHub data to refine this questions later. "
    "You will need to pass an OpenAI API key to continue:"
)

openai_api_key = st.text_input("OpenAI API Key", type="password")
client = OpenAI(api_key=openai_api_key)
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
prompt = st.chat_input("Ask questions about companies who use cudf")
if prompt:
    # if prompt := st.chat_input("Ask questions about companies who use cudf"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})


st.markdown(
    "We can explore the location of users who post on cudf. "
    "This can help with event planning and community building. "
    "The legend isn't showing up where yellow marker are NVIDIA employees "
    "and blue markers are non-NVIDIA employees. See the plots.ipynb file for a clean version of this"
)

gdf_community = gpd.GeoDataFrame(
    df_community,
    geometry=gpd.points_from_xy(
        df_community["location_lon"], df_community["location_lat"]
    ),
    crs="epsg:4326",
)

m = gdf_community[
    ["login", "name", "followers", "company", "is_nvidia_employee", "geometry"]
].explore(column="is_nvidia_employee", cmap="viridis")
st_folium(m, width=725)

st.markdown(
    "We can now query the GitHub cudf dataset. "
    "Here we can ask questions such as 'Which issues has the most thumbs up?' "
    "or 'What are users from walmart interested in?' "
)
