from huggingface_hub import login
from pyngrok import ngrok
import os
from pyngrok import conf
from huggingface_hub import login


os.environ["HUGGINGFACEHUB_API_TOKEN"]="<hf_token>"
login(os.environ["HUGGINGFACEHUB_API_TOKEN"])

conf.get_default().auth_token = "<ngrok_token>"

%cd project1_2_1

# Kill any existing tunnels
ngrok.kill()

# Run Streamlit in background
# os.system("streamlit run main.py &")
os.system("streamlit run main_with_context.py &")

# Open Ngrok tunnel
public_url = ngrok.connect(8501)
print(f"Open your chatbot here: {public_url}")
