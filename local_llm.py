from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM

ollama_llm = LocalLLM(api_base="http://localhost:11434/v1", model="mistral-nemo")
df = SmartDataframe("all_stocks_5yr.csv", config={"llm": ollama_llm})

response = df.chat("What was the average open for INTC in 2016")
print(response)