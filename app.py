import os

# conda install -c conda-forge 'ffmpeg<7'
os.system("apt install ffmpeg")
os.system("apt install libsox-dev")
os.system('streamlit run demo/app.py --server.port 7860')
