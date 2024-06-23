# LLM_Bedrock
This is a end-to-end RAG with Claude and Llama 2 models from AWS bedrock.

#install
conda create -p venv python==3.10 -y
conda activate venv/
pip install -r requirements.txt


#aws configure (create IAM first in AWS , create key then and save in csv)
in VS terminal
>> aws configure
>> enter Id
>> enter secret access key
>> enter region -> us-east-1
>> enter format -> json

#run the app
streamlit run llm_app.py
