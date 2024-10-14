from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM

ollama_llm = LocalLLM(api_base="http://localhost:11434/v1", model="mistral-nemo", temperature=0)
sdf = SmartDataframe("all_stocks_5yr.csv", config={"llm":ollama_llm})

response = sdf.chat("What was the average open for INTC in september of 2016?")
print(response)