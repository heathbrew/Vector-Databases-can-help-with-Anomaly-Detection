# for nvidia gpus on windows only 
python -m venv ./venv
./venv/Scripts/activate
python -m pip install --upgrade pip
pip install dataclasses
pip install jupyterlab
# installing some general liabraries
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install scipy
# for dataframe and some error
pip install pandas
pip install pyarrow
# for xlsx conversion
pip install openpyxl
# installing tensorflow
pip install tensorflow
pip install tensorflow-addons
# installing pytorch for cuda
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# making enviornment for llm from hugging face
$env:FORCE_CMAKE=1
$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
pip install llama-index
pip install transformers
pip install torch
pip install langchain-community
pip install langchain
# installing faiss
pip install faiss-cpu  # For CPU
# installing quadrant client
pip install --upgrade llama-index-core


## scrapping
pip install requests
pip install bs4
pip install lxml
pip install sentence-transformers
pip install pinecone-client
pip install seaborn




