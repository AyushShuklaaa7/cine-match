import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

# 1. Page Configuration
st.set_page_config(
    page_title="AI Movie Concierge", 
    page_icon="🎬", 
    layout="wide"
)

# Application Header
st.title("🎬 AI Personal Entertainment Concierge")
st.markdown("Discover highly contextual recommendations powered by Natural Language Processing.")
st.markdown("---")

# 2. Optimized Data Loading with Caching
@st.cache_data
def load_data():
    # Reads the preprocessed data generated in Google Colab
    df = pd.read_csv('movies_cleaned.csv')
    # Fill any accidental missing textual values to prevent vectorizer breaks
    df['metadata_soup'] = df['metadata_soup'].fillna('')
    df['overview'] = df['overview'].fillna('No summary available.')
    df['genres'] = df['genres'].fillna('Uncategorized')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ **Data File Missing:** 'movies_cleaned.csv' was not found in the directory. Please ensure it is uploaded alongside this script.")
    st.stop()

# 3. Vectorization & Similarity Computations
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['metadata_soup'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a clean string-to-index mapping reference table
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# 4. User Configuration Interface (Sidebar)
st.sidebar.header("👤 Your Taste Profile")
st.sidebar.write("Search and select the movies or series you have thoroughly enjoyed:")

watched_movies = st.sidebar.multiselect(
    label="Search Watch History:",
    options=df['title'].values,
    default=None,
    placeholder="Type to filter titles..."
)

# 5. Core Application Dashboard Framework
tab1, tab2 = st.tabs(["🎬 Smart Recommendations", "🔍 Entertainment Encyclopedia"])

# --- TAB 1: RECOMMENDATIONS LOGIC & INTERFACE ---
with tab1:
    if not watched_movies:
        st.info("👈 **Get Started:** Select one or multiple titles in the left sidebar menu to calculate your real-time recommendations!")
    else:
        st.subheader("🔥 Top Recommendations Matched to Your Profile")
        
        # Aggregate taste vectors across multiple selections
        selected_indices = [indices[movie] for movie in watched_movies if movie in indices]
        
        if selected_indices:
            # Extract scores and calculate their mathematical average to establish a single target profile
            mean_sim_scores = cosine_sim[selected_indices].mean(axis=0)
            
            # Sort the database entries based on highest similarity correlation
            sorted_movie_indices = np.argsort(mean_sim_scores)[::-1]
            
            # Filter the outputs to exclude movies already listed in the history setup
            final_recommendations = []
            for idx in sorted_movie_indices:
                title = df.iloc[idx]['title']
                if title not in watched_movies:
                    final_recommendations.append(idx)
                if len(final_recommendations) == 5: # Cap recommendation cards at 5
                    break
            
            # Render structured graphical UI containers for each recommendation
            for idx in final_recommendations:
                rec_title = df.iloc[idx]['title']
                rec_genres = df.iloc[idx]['genres']
                rec_overview = df.iloc[idx]['overview']
                
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"### {rec_title}")
                        st.caption(f"🎭 **Genres:** {rec_genres}")
                        st.write(f"{rec_overview}")
                    
                    with col2:
                        st.write("")  # Strategic spacing layout buffer
                        search_query = urllib.parse.quote(f"where to watch {rec_title} streaming India")
                        google_search_url = f"https://www.google.com/search?q={search_query}"
                        
                        st.link_button(
                            label="📺 Check Platforms (India)",
                            url=google_search_url,
                            use_container_width=True
                        )

# --- TAB 2: ENCYCLOPEDIA LOOKUP ---
with tab2:
    st.subheader("🔍 Movie Encyclopedia Lookup")
    st.write("Look up complete text profiles, system tokens, and indexing signatures inside the local AI database.")
    
    search_selection = st.selectbox(
        label="Select a title to view details:", 
        options=df['title'].values,
        index=0
    )
    
    if search_selection:
        matched_row = df[df['title'] == search_selection].iloc[0]
        
        st.markdown(f"## {matched_row['title']}")
        st.markdown(f"**🎭 System Genre Classifications:** `{matched_row['genres']}`")
        
        st.markdown("### 📖 Original Plot Synopsis")
        st.write(matched_row['overview'])
        
        st.markdown("### 🧪 Vector Engine Metadata Soup")
        st.caption("This is the exact token signature analyzed by the NLP model to compute cross-movie similarity structures:")
        st.info(matched_row['metadata_soup'])"
