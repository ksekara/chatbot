# run_streamlit_app.py
import subprocess

# Path to your Streamlit app
streamlit_app_path = 'interface.py'

# Run the Streamlit app
subprocess.run(['streamlit', 'run', streamlit_app_path])
