from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM

ollama = LocalLLM(api_base="http://localhost:11434/v1", model='codellama')
df = SmartDataframe('all_stocks_5yr.csv', config={'llm':ollama})

response = df.chat('graph a line plot of the average volume for INTC and AAL for every month in 2015')
print(response)