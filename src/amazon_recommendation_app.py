import streamlit as st
import pandas as pd
import os

top_5_rated_per_user = pd.read_csv(os.path.join("..", 'user_item_top_5_recommendations_by_ratings.csv'))
most_popular_items_per_category = pd.read_csv(os.path.join("..", 'most_popular_items_per_category.csv'))


# Define a function to simulate fetching top 5 recommendations
def move_cols_to_first(the_df, first_cols):
    """
    Rearrange df columns so that first_cols will be first
    :param the_df: A dataframe
    :param first_cols: a list of columns to be first cols in the df
    :return: the dataframe rearranged
    """
    the_df = pd.concat([the_df[first_cols], the_df.loc[:, ~the_df.columns.isin(first_cols)]], axis=1)
    return the_df

def get_item_recommendations_for_userName(userName):
    recommended_items = top_5_rated_per_user.loc[top_5_rated_per_user['userName'] == userName, 'item_id'].str.split('_', expand=True)
    if len(recommended_items) > 0:
        recommended_items.columns = ['brand','itemName', 'price']
        recommended_items = move_cols_to_first(recommended_items, ['itemName'])
        recommended_items = recommended_items.reset_index(drop=True)
    else:
        recommended_items = most_popular_items_per_category

    return recommended_items


# Function to display recommendations in a fancy way
def display_recommendations(recommendations):
    # Use a beta expander to hide the recommendations by default
    with st.expander("See your recommendations"):
        # Display each recommendation in a visually appealing format
        st.table(recommendations)

# Set the page layout to wide mode for better space utilization
st.set_page_config(layout="wide")

# Custom Streamlit theme options
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Title of the app
#st.markdown('<p class="big-font">User Item Recommendation System</p>', unsafe_allow_html=True)
for i in range(5):
    st.write("")

# Side bar for user input
with st.sidebar:
    st.image("data:image/png;base64,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")
    st.write("# How does it work?")
    # Button to get recommendations
    st.video(r'https://www.youtube.com/watch?v=S4RL6prqtGQ')

# Main container
with st.container():
    col1, col2 = st.columns([1, 2])  # Split the main area into 1:2 ratio

    with col1:  # Column for user input
        st.write("## Your Top Amazon Recommendations")
        user_name = st.text_input('Enter your username', '')
        button_recommendation = st.empty()
        if button_recommendation.button('Get Recommendations'):
            if user_name:
                # Fetch the top 5 recommendations for the user
                top_recommendations = get_item_recommendations_for_userName(user_name)
                # Display recommendations in the second column
                display_recommendations(top_recommendations)
            else:
                st.warning('Please enter a username.')


    with col2:  # Column for displaying recommendations
        st.image("https://assets-global.website-files.com/60bfd2b558c9eba77e06bf57/60f591a8e727a820ea67615d_amazon-recos-feat.png")  # Replace with your actual logo
        #st.image("https://m.media-amazon.com/images/G/01/gc/designs/livepreview/amazon_a_black_noto_email_v2016_us-main._CB624175556_.png", width=150)  # Replace with your actual logo
        # Placeholder for recommendations
        placeholder = st.empty()


# Note: You can move the button and recommendation display logic outside the `with` block
# if you want the recommendations to appear below the input column.

