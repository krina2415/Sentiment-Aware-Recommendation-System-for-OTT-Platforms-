import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from hybrid_system import HybridRecommenderSystem

# Page configuration
st.set_page_config(
    page_title="Sentiment-Aware Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

class RecommenderApp:
    def __init__(self):
        self.system = None
        self.initialized = False
    
    def initialize_system(self):
        """Initialize the recommender system."""
        with st.spinner("Loading data and training models..."):
            try:
                preprocessor = DataPreprocessor()
                
                # Load IMDB data for sentiment
                imdb_data = preprocessor.load_imdb_data()
                
                # Load movies data
                movies, ratings = preprocessor.load_movies_data()
                
                # Create sentiment features
                movies = preprocessor.create_movie_sentiment_features(movies, imdb_data)
                
                # Initialize hybrid system
                self.system = HybridRecommenderSystem()
                self.system.initialize_system(movies, ratings)
                
                self.initialized = True
                st.success("System initialized successfully!")
                
            except Exception as e:
                st.error(f"Error initializing system: {str(e)}")
    
    def run(self):
        """Run the Streamlit application."""
        st.markdown('<h1 class="main-header">🎬 Sentiment-Aware Movie Recommender</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/3160/3160576.png", 
                    width=100)
            st.title("Settings")
            
            # Initialize button
            if st.button("Initialize System", type="primary"):
                self.initialize_system()
            
            st.markdown("---")
            
            if self.initialized:
                # User input
                user_id = st.number_input(
                    "Enter User ID", 
                    min_value=1, 
                    max_value=10000, 
                    value=1
                )
                
                num_recommendations = st.slider(
                    "Number of Recommendations", 
                    min_value=5, 
                    max_value=20, 
                    value=10
                )
                
                # Recommendation type
                rec_type = st.selectbox(
                    "Recommendation Type",
                    ["Sentiment-Aware", "Collaborative Filtering", "Diverse"]
                )
                
                st.markdown("---")
                st.info("🎯 Sentiment-aware recommendations consider both ratings and emotional tone of movies.")
            
        # Main content
        if self.initialized:
            # User analysis section
            st.header(f"📊 User {int(user_id)} Profile Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                user_analysis = self.system.analyze_user_preferences(int(user_id))
                
                if "error" not in user_analysis:
                    st.metric("Average Rating", f"{user_analysis['average_rating']:.2f}")
            
            with col2:
                if "error" not in user_analysis:
                    st.metric("Total Ratings", user_analysis['total_ratings'])
            
            with col3:
                if "error" not in user_analysis:
                    st.metric("Profile Type", user_analysis['sentiment_profile'].replace("_", " ").title())
            
            # Top genres visualization
            if "error" not in user_analysis and user_analysis['top_genres']:
                genres_df = pd.DataFrame(
                    user_analysis['top_genres'], 
                    columns=['Genre', 'Count']
                )
                
                fig = px.bar(
                    genres_df, 
                    x='Genre', 
                    y='Count',
                    title="Top Genres Preferred",
                    color='Count',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Get recommendations
            st.header("🎯 Recommended Movies")
            
            if rec_type == "Sentiment-Aware":
                recommendations = self.system.get_sentiment_aware_recommendations(
                    int(user_id), num_recommendations
                )
            elif rec_type == "Collaborative Filtering":
                recommendations = self.system.recommender.get_recommendations(
                    int(user_id), num_recommendations
                )
            else:
                recommendations = self.system.get_diverse_recommendations(
                    int(user_id), num_recommendations
                )
            
            if not recommendations.empty:
                # Display recommendations
                for idx, row in recommendations.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.subheader(f"{idx + 1}. {row['title']}")
                        
                        with col2:
                            score = row.get('adjusted_score', row.get('predicted_rating', 0))
                            st.metric("Score", f"{score:.2f}")
                        
                        # Show movie details if available
                        movie_info = self.system.movies_df[
                            self.system.movies_df['id'] == row['movieId']
                        ]
                        
                        if not movie_info.empty:
                            with st.expander("Movie Details"):
                                info = movie_info.iloc[0]
                                
                                if pd.notna(info.get('overview')):
                                    st.write("**Overview:**", info['overview'][:200] + "...")
                                
                                if pd.notna(info.get('genres')):
                                    st.write("**Genres:**", info['genres'])
                                
                                if pd.notna(info.get('sentiment_score')):
                                    sentiment = "😊 Positive" if info['sentiment_score'] > 0.6 else \
                                               "😐 Neutral" if info['sentiment_score'] > 0.4 else \
                                               "😞 Negative"
                                    st.write(f"**Sentiment Profile:** {sentiment}")
                        
                        st.markdown("---")
            else:
                st.warning("No recommendations available. Try a different user ID.")
            
            # System statistics
            st.header("📈 System Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Movies", len(self.system.movies_df))
            
            with col2:
                st.metric("Total Users", len(self.system.user_profiles))
            
            with col3:
                avg_ratings = self.system.ratings_df['rating'].mean()
                st.metric("Average Rating", f"{avg_ratings:.2f}")
        
        else:
            # Welcome screen
            st.info("👈 Click 'Initialize System' in the sidebar to get started!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 🎯 Features
                - **Sentiment Analysis**: Understand movie emotional tones
                - **Collaborative Filtering**: Learn from user preferences
                - **Hybrid Approach**: Best of both worlds
                - **Personalized**: Tailored to each user's taste
                """)
            
            with col2:
                st.markdown("""
                ### 📊 Data Sources
                - **IMDB 50K Reviews**: For sentiment training
                - **Movies Dataset**: 26M ratings, 45K movies
                - **270K Users**: Real user behavior patterns
                """)
            
            st.markdown("---")
            
            # Sample data preview
            st.subheader("📁 Sample Data Preview")
            
            try:
                preprocessor = DataPreprocessor()
                movies, ratings = preprocessor.load_movies_data()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Movies Metadata (Sample)**")
                    st.dataframe(movies.head(10), use_container_width=True)
                
                with col2:
                    st.write("**Ratings Data (Sample)**")
                    st.dataframe(ratings.head(10), use_container_width=True)
            except:
                pass

if __name__ == "__main__":
    app = RecommenderApp()
    app.run()