import streamlit as st
from crew import create_research_crew
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# Header section
st.title("🤖 AI Research Assistant")
st.markdown("""
    This AI-powered research assistant can help you gather and analyze information on any topic.
    Simply enter your research topic below and let our AI crew do the work for you!
    """)

# Main content
with st.container():
    topic = st.text_input("Enter your research topic:", 
                         placeholder="e.g., Impact of AI on Healthcare")
    
    if st.button("Start Research", type="primary") and topic:
        with st.spinner("Research in progress..."):
            try:
                crew = create_research_crew(topic)
                result = crew.kickoff()
                
                st.markdown("## Research Results")
                st.markdown(result)
                
                st.download_button(
                    label="Download Report",
                    data=result,
                    file_name=f"research_report_{topic.lower().replace(' ', '_')}.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}") 