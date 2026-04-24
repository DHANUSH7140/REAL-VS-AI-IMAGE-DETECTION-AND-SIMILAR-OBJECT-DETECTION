"""
shared/theme.py — Theme CSS for the Streamlit Unified Router.
"""

def get_theme_css(dark_mode: bool) -> str:
    """Returns the CSS string for the requested theme mode, hiding the deploy menu and upgrading the UI."""
    
    # Common CSS to hide Streamlit header, footer, and deploy menu
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Glassmorphism Container */
        .glass-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 3rem;
            margin: auto;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            max-width: 800px;
        }
        
        /* Premium Buttons */
        .stButton > button {
            border-radius: 12px !important;
            padding: 15px 30px !important;
            font-weight: 800 !important;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            width: 100% !important;
            height: 70px !important;
            font-size: 1.25rem !important;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .stButton > button:hover {
            transform: translateY(-4px) scale(1.02);
        }
        </style>
    """

    if dark_mode:
        return hide_streamlit_style + """
        <style>
            .stApp {
                background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
                color: #f8fafc;
            }
            .stButton > button {
                background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
                color: #ffffff !important;
                border: none !important;
                box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
            }
            .stButton > button:hover {
                box-shadow: 0 8px 25px rgba(124, 58, 237, 0.6);
            }
            h1 {
                font-family: 'Inter', sans-serif;
                font-weight: 900;
                font-size: 3.5rem !important;
                background: linear-gradient(to right, #a78bfa, #f472b6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 1rem;
            }
            p.subtitle {
                color: #94a3b8;
                font-size: 1.3rem;
                margin-bottom: 3rem;
                font-weight: 500;
            }
            /* Theme Toggle Button Special Styling */
            button[data-testid="baseButton-secondary"] {
                background: rgba(255, 255, 255, 0.1) !important;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(255, 255, 255, 0.2) !important;
                color: white !important;
                height: 40px !important;
                font-size: 0.9rem !important;
                width: auto !important;
                padding: 0 15px !important;
            }
        </style>
        """
    else:
        return hide_streamlit_style + """
        <style>
            .stApp {
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                color: #0f172a;
            }
            .stButton > button {
                background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
                color: #ffffff !important;
                border: none !important;
                box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
            }
            .stButton > button:hover {
                box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5);
            }
            h1 {
                font-family: 'Inter', sans-serif;
                font-weight: 900;
                font-size: 3.5rem !important;
                background: linear-gradient(to right, #2563eb, #db2777);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 1rem;
            }
            p.subtitle {
                color: #475569;
                font-size: 1.3rem;
                margin-bottom: 3rem;
                font-weight: 500;
            }
            /* Theme Toggle Button Special Styling */
            button[data-testid="baseButton-secondary"] {
                background: rgba(0, 0, 0, 0.05) !important;
                border: 1px solid rgba(0, 0, 0, 0.1) !important;
                color: #0f172a !important;
                height: 40px !important;
                font-size: 0.9rem !important;
                width: auto !important;
                padding: 0 15px !important;
            }
        </style>
        """
