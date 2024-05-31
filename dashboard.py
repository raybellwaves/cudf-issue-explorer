"""
streamlit run dashboard.py
"""

from openai import OpenAI
from langchain.agents import AgentType
from streamlit_folium import st_folium
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import OpenAI as OpenAI_langchain
from langchain_community.callbacks import StreamlitCallbackHandler
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.title("cudf GitHub issue explorer for DevRel")

st.markdown(
    """
This dashboard can help with:
- Identification of users/leads
- Identify common developer pain points
- Identify most common requested features
"""
)

st.markdown(
    "**This dashboard is WIP and may be prone to errors.** "
    "Code associated with this can be found at "
    "https://github.com/raybellwaves/cudf-issue-explorer"
)

st.subheader("Partners")

st.markdown(
    "We can explore what companies are active in in the cudf GitHub repo. "
    "These are likely super users:"
)


@st.cache_data
def fetch_community_data():
    return pd.read_parquet("all_poster_commenter_details.parquet")


@st.cache_data
def fetch_issue_data():
    return pd.read_parquet("external_issue_details_with_posters_min.parquet")


def chat_response(content):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content


def agent_response(agent, content):
    return agent.invoke(content)["output"]


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
    "To learn more about these companies we will ask a LLM questions such as "
    "**'What type of company is Halliburton?'** "
    "or generic questions such as "
    "**'Why would Halliburton use GPUs to speed up data processing?'** "
    "but don't expect a good result. "
    "We will use the GitHub data to refine this question later. "
)

st.markdown("**You will need to pass an OpenAI API key to ask questions below:**")

openai_api_key = st.text_input("OpenAI API Key:", type="password")
client = OpenAI(api_key=openai_api_key)


content = st.text_input(
    "Ask questions about companies who use cudf:",
    "What type of company is Halliburton?",
)
if openai_api_key:
    st.write(chat_response(content))
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
# prompt = st.chat_input("Ask questions about companies who use cudf")
# if prompt:
#     # if prompt := st.chat_input("Ask questions about companies who use cudf"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     with st.chat_message("assistant"):
#         stream = client.chat.completions.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ],
#             stream=True,
#         )
#         response = st.write_stream(stream)
#     st.session_state.messages.append({"role": "assistant", "content": response})


st.subheader("Community building")

st.markdown(
    "We can explore the location of users who post on cudf. "
    "This can help with event planning and community building. "
    "The legend isn't showing up (https://github.com/randyzwitch/streamlit-folium/issues/192) on this map where yellow marker are NVIDIA employees "
    "and blue markers are non-NVIDIA employees. See https://github.com/raybellwaves/cudf-issue-explorer/blob/main/plots.ipynb for a clean version of this."
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
].explore(
    column="is_nvidia_employee",
    cmap="viridis",
    legend=True,
)
st_folium(m, width=1000)

st.subheader("Understanding developers")

st.markdown(
    "We can explore the GitHub data to understand what developers are interested in "
    "and to ensure their requested features or bug are taken into account in the roadmap"
    "You can ask questions such as "
    "**What issues are Walmart most interested in?**"
    "**What issue is ETH Zürich most interested in?**"
    "**What issue has the most thumbs up?**"
    "**What company posted the issue with the most thumbs up?**"
    "**What are the top 5 issues with the most thumbs up?**"
)
st.markdown("**Note: this is a subsample of the data for demo puposes**")

df_issues = fetch_issue_data().drop(columns="body")
agent = create_pandas_dataframe_agent(
    OpenAI_langchain(
        temperature=0,
        model="gpt-3.5-turbo-instruct",
        openai_api_key=openai_api_key,
    ),
    df_issues,
)
content = st.text_input(
    "Ask questions about about external cudf users and developers using the GitHub data:",
    "What issues are Walmart most interested in?",
)
if openai_api_key:
    st.write(agent_response(agent, content))

# if "messages" not in st.session_state or st.sidebar.button(
#     "Clear conversation history"
# ):
#     st.session_state["messages"] = [
#         {
#             "role": "assistant",
#             "content": "How can I help you answer questions about external cudf users and developers?",
#         }
#     ]
# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])
# if prompt := st.chat_input(placeholder="What is this data about?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)
#     if not openai_api_key:
#         st.info("Please add your OpenAI API key to continue.")
#         st.stop()
#     llm = OpenAI_langchain(
#         temperature=0,
#         model="gpt-3.5-turbo-instruct",
#         openai_api_key=openai_api_key,
#         streaming=True,
#     )
#     pandas_df_agent = create_pandas_dataframe_agent(
#         llm,
#         df_issues,
#     )
#     with st.chat_message("assistant"):
#         st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
#         response = pandas_df_agent.invoke(st.session_state.messages, callbacks=[st_cb])
#         st.session_state.messages.append({"role": "assistant", "content": response})
#         st.write(response)
