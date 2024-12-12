from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_openai import ChatOpenAI
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,PromptTemplate
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
import time
def generate_followup_qns(context):
        print("INITIATING FOLLOW UP QN",time.time() * 1000)

        out_format = """the output should be confined within a Proper json structure as following:
                        {"questions": list//the list of 3 relevant questions from the context}"""
        
        ex_out = """the example output is: 
                {
                        "questions": [
                            "question 1",
                            "question 2",
                            "question 3"
                        ]
                    }"""

        client = OpenAI(api_key = OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "user", "content": f"""You are a AI tool who can provide 3 follow up questions based on the context given to you.
                Let the follow up questions be clearly demarcated and numbered
                
                {out_format}

                {ex_out}

                context:
                {context}\n\n
                
                Output:            
                """},
                #{"role": "user", "content": f"""{context_question}\n\n{out_format}"""},
            ],
            max_tokens=4000,
            n=1,
            temperature=0,
        )
        parser = JsonOutputParser()
        parsed_resp = parser.parse(response.choices[0].message.content)
        print(" FOLLOW UP QN COMPLETED",time.time() * 1000)

        return parsed_resp

