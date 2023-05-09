python -m venv venv
.\venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
.\venv\Scripts\pip install -r .\requirements.txt
.\venv\Scripts\pip install -e .
.\venv\Scripts\activate
