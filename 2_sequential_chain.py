from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

llm1 = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    timeout=30,
    max_retries=2,
    temperature= 0.7
)

llm2 = ChatOpenAI(
    model="gpt-4o",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    timeout=30,
    max_retries=2,
    temperature= 0.5
)

parser = StrOutputParser()

chain = prompt1 | llm1| parser | prompt2 | llm2 | parser

config = {
    'tags' : ['llm_app', 'report_generation', 'summarization'],
    'metadata' : {'llm1' : 'gpt-4o-mini','llm1_temp' :0.7}
}

result = chain.invoke({'topic': 'Unemployment in India'}, config = config)

print(result)
