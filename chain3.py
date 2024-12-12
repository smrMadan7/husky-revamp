import os
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import StuffDocumentsChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


from langchain.retrievers.document_compressors import FlashrankRerank
from openai import OpenAI
import re
import copy
import followup_qns
from dotenv import load_dotenv
from database import Storage
from text_citation import StuffDocumentsWithIndexChain
load_dotenv()
# from rerouter import RerouteData
import traceback
from neo4j.time import DateTime as Neo4jDateTime
from datetime import datetime, date, timedelta
client = OpenAI()
import time

class Chatbot:
    print(" CHATBOT INITIATED",time.time() * 1000)

    combine_doc_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""Your role is to provide detailed information based on the following sources.
    When referencing the documents, add a citation right after. Use "[SOURCE_NUMBER](source_url)" for the citation (e.g. "The Space Needle is in Seattle [1](https://www.space.com)[2](https://www.astro_space.com).")

    Sources:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}"""
    )

    combine_docs_chain = StuffDocumentsWithIndexChain(
        llm_chain=LLMChain(
            llm=ChatOpenAI(model=os.getenv('retrieval_model'), max_tokens=int(os.getenv('max_tokens')),
                           temperature=float(os.getenv("temperature"))),
            prompt=combine_doc_prompt,
        ),
        document_prompt=PromptTemplate(
            input_variables=["index", "source", "page_content"],
            template="[{index}] {source}:\n{page_content}",
        ),
        document_variable_name="context",
    )
    temperature=float(os.getenv("temperature"))

    def __init__(self, chat_history=None):
        self.temperature = Chatbot.temperature
        # print(self.temperature)
        self.model = os.getenv("model")
        self.retrieval_model = os.getenv("retrieval_model")
        self.streaming = bool(os.getenv("streaming"))
        self.no_matches = int(os.getenv("no_matches"))
        self.store = Storage().connect_storage()
        self.store.index_name='PL_Twitter_Large'
        self.store.node_label='Twitter_Chunk'
        self.no_rerank_documents = int(os.getenv("no_rerank_documents"))
        self.max_tokens = int(os.getenv("max_tokens"))
        self.retrieval_chain = self.create_retrieval_chain()
        self.chat_history = chat_history if chat_history is not None else []
        self.summarizer=os.getenv('summarizing_model')
    def format_document(self, doc, index, prompt, chat_history):
        base_info = {"page_content": doc.page_content, "index": index, "source": doc.metadata['source']}
        missing_metadata = set(prompt.input_variables).difference(base_info)
        if len(missing_metadata) > 0:
            raise ValueError(f"Missing metadata: {list(missing_metadata)}.")
        document_info = {k: base_info[k] for k in prompt.input_variables}
        document_info["chat_history"] = chat_history  # Add chat history to document info
        return prompt.format(**document_info)

    #from twitter bot
    def summarize_text(self,text):
        print(" SUMMARY INITIATED",time.time() * 1000)

        chat_model = ChatOpenAI(model=self.summarizer)
        prompt_template = PromptTemplate(
            input_variables=["text"],
            template="""
                Please summarize the following text in meaningful manner. Please output only 2 lines of information as summary:
                {text}
                """
        )

        # Create an LLMChain with the chat model and the prompt template
        chain = LLMChain(llm=chat_model, prompt=prompt_template)

        # Generate the summary
        response = chain.run({"text": text})

        return response.strip()

    def get_answer(self, query, chat_history, field_building,substack_check):
        print(" RETRIEVAL INITIATED",time.time() * 1000)

        # Append the user query to the chat history
        chat_history.append({"role": "user", "content": query})

        # Generate a standalone question from the user query and chat history
        question = self.create_standalone_question(query, chat_history)
        if field_building or substack_check :
            print("labweek")
            self.store.index_name='PL_Substack'
            self.store.node_label='Substack_latest'
     # Retrieve context documents based on the standalone question and chat history
        context_docs = self.retrieve_context(question, chat_history,substack_check)
        # print(context_docs)
        # print("context_docs")
        # # Format the context documents
        # context = self.format_context(context_docs)

        # Generate a response using the context, question, and context documents
        context_question = {
            "context": context_docs,
            "chat_history": chat_history,
            "question": question,
        }
        print("INITIATING RESPONSE",time.time() * 1000)

        response = self.generate_response(context_question, context_docs)
        print("RESPONSE OBTAINED",time.time() * 1000)

        if "error" in response :
            return response
        else :
            answer, references, follow_up_questions,filtered_title,filtered_description = response

        # Append the model's response to the chat history
        chat_history.append({"role": "assistant", "content": answer})
        """reroute_instance = RerouteData()
        entities=reroute_instance.extract_entities(answer,references)
        self.store.index_name="PL_Twitter_Large"
        self.store.node_label="Twitter_Chunk"
        all_values = list(entities.values())
        flattened_values = [item for sublist in all_values for item in sublist]
        unique_values = list(set(flattened_values))
        # print("Entities Present in Response ------------------>",unique_values)
        queries_list,results_list = reroute_instance.process_queries_concurrently(unique_values)
        print("additional information for rerouting----------------->",results_list)
        summary_data=[]
        if results_list:
            for entry in results_list:
                for key in entry:
                    for item in entry[key]:
                        item['summary'] = self.summarize_text(item['text'])
                        summary_data.append((item['summary'],item['metadata']))
            summary_enumerated = [(index, item) for index, item in enumerate(summary_data,start=1)]
        else:
            summary_enumerated = None
        summary_enumerated"""
        sources = self.format_source(references,filtered_title,filtered_description)

        return answer, sources, follow_up_questions,references#,summary_enumerated
    def format_source(self,references,filtered_title,filtered_description):
        print("INITIATING SOURCE FORMATTING",time.time() * 1000)

        source_dict = {
            "id": "",
            "link": "",
            "title": "",
            "description": ""
        }

        final_source = []
        for i in range(0,int(len(references))):
            source_dict["id"] = i+1
            source_dict["link"] = references[i][1]
            source_dict["title"] = filtered_title[i][1]
            source_dict["description"] = filtered_description[i][1]
            final_source = final_source + [copy.deepcopy(source_dict)]
        final_source = [entry for entry in final_source if not (entry["link"] == "None" or entry["title"] == "None")]
        for index, entry in enumerate(final_source, start=1):
            entry["id"] = index
        # print(final_source)
        # print("final_source")
        print("RETURN SOURCE FORMATTING",time.time() * 1000)

        return final_source

    # class Chatbot:
#     def __init__(self, chat_history=None):
#         self.temperature = float(os.getenv("temperature"))
#         self.model = os.getenv("model")
#         self.retrieval_model = os.getenv("retrieval_model")
#         self.streaming = bool(os.getenv("streaming"))
#         self.no_matches = int(os.getenv("no_matches"))
#         self.store = storage().connect_storage()
#         self.no_rerank_documents = int(os.getenv("no_rerank_documents"))
#         self.max_tokens = int(os.getenv("max_tokens"))
#         self.retrieval_chain = self.create_retrieval_chain()
#         self.chat_history = chat_history if chat_history is not None else []

    def create_standalone_question(self, question, chat_history):
        model = ChatOpenAI()

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "question - {question}, chat history - {chat_history}"),
            ]
        )

        contextualize_q_chain = contextualize_q_prompt | model | StrOutputParser()
        response = contextualize_q_chain.invoke({"question": question, "chat_history": chat_history})

        if isinstance(response, dict) and 'text' in response:
            return response['text']
        else:
            return question

    import re
    def validate_presense_pers_info(self,message):
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        phone_pattern = r'(?<!\S)(\+?\d{1,4}[\s-]?)?(?!0+\s*,?$)(\d{10,14})(?!\S)'

        emails = re.findall(email_pattern, message)
        phone_numbers = re.findall(phone_pattern, message)

        # Extracting the phone number from the matched groups
        phone_numbers = [match[1] for match in phone_numbers]


        # print(emails)
        # print(phone_numbers)
        if len(emails) > 0 or len(phone_numbers)>0:
            return True
        else:
            return False
    #validate_presense_pers_info(message)
    def mask_personal_info(self, message):
      print(" MASKING  BEGINS",time.time() * 1000)

      try:
            model = ChatOpenAI(temperature = 0)

            mask_pers_info_systemprompt = """Given a text, rephrase it to remove an individuals contact information such as email addresses while preserving all other information, including organization emails, names and citations. Ensure that the sentence remains complete and grammatically correct, and any trailing references,citations or footnotes should be preserved. If no individual email is found, return the text unchanged.
                                            Input:
                                                  The Lotus team in tentin is heavily involved in the development.
                                                  You can contact the organization directly at marketing@tentin.in.[1]
                                                  For more information on this context you can contact Mr. Manen, one of the founders of tentin.He can be contacted via email at Manen@tentin.in [2].
                                            Expected Output:
                                            The Lotus team in tentin is heavily involved in the development.
                                            You can contact the organization directly at marketing@tentin.in.[1]
                                            For more information on this context you can contact Mr. Manen, one of the founders of tentin [2].
                                            """

            mask_pers_info_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", mask_pers_info_systemprompt),
                    ("human", "input text - {message}\n\noutput text -"),
                ]
            )

            mask_pers_info_chain = mask_pers_info_prompt | model | StrOutputParser()
            masked_message = mask_pers_info_chain.invoke({"message": message})
            # print(masked_message)
            #print("hi")
            print(" MASKING ENDS",time.time() * 1000)

            return {"masked_message":masked_message}
            
      except Exception as e:
            return {"message":"unable to find or mask the personal information","error":str(e)}
            
                
    def create_retrieval_chain(self):

        compressor = FlashrankRerank(top_n=self.no_rerank_documents)
        store = self.store
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=store.as_retriever(search_kwargs={"k": self.no_matches}, return_source_documents=True)
        )
        chat_model = ChatOpenAI(temperature=self.temperature, model=self.retrieval_model, max_tokens=self.max_tokens,
                                streaming=self.streaming)

        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="""Please answer the question with citation to the paragraphs. For every sentence you write, cite the source name and reference number provided. 
                     Provide answers in a structured way with an introduction, main content with sub-sections, and a conclusion.
                  Context: {context}
                  Question: {query}
                  Answer: """
        )

        return RetrievalQA.from_chain_type(
            llm=chat_model,
            retriever=compression_retriever,
            chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"},
            return_source_documents=True
        )
    import traceback
    def generate_response(self, context_question, context_docs):
        try:
            print(" INITIALTING RESP INNER LOOP",time.time() * 1000)

            response = self.combine_docs_chain.invoke(
                {
                    "context": context_question["context"],
                    "chat_history": context_question["chat_history"],
                    "question": context_question["question"],
                    "input_documents": context_docs
                }
            )
            print(" UNPARSED INNER LOOP RESPONSE",time.time() * 1000)

        except Exception as e:
            #print(traceback.format_exc())
            return {"message":"error retrieving content from Openai","error": str(e)}
        if response and response["output_text"]:
            print(" CITATATION BEGINS",time.time() * 1000)
            annotated_response = response["output_text"]
            references = "\n\nReferences:\n"
            title = "\n\ntitle:\n"
            description = "\n\ndescription: \n"
            # print(context_docs)
            for i, doc in enumerate(context_docs):
                #print(doc)
                references += f"{i + 1}. {doc.metadata['source']}\n"
                if 'title' in doc.metadata:
                   title += f"{i + 1}. {doc.metadata['title']}\n"
                else:
                   title += f"{i + 1}. {None}\n"
                if 'description' in doc.metadata:
                   description += f"{i + 1}. {doc.metadata['description']}\n"
                else:
                   description += f"{i + 1}. {None}\n"



            
            # print(title)
            # print(references)
            cited_reference = self.extract_cited_references(annotated_response)
            # print("cited references------------->", cited_reference)
            filtered_references = self.filter_references_to_keep(references, cited_reference)
            filtered_title = self.filter_titles_to_keep(title, cited_reference)
            filtered_description = self.filter_titles_to_keep(description, cited_reference)

            updated_response, reorganized_references,reorganized_title,reorganized_description = self.renumber_references(annotated_response, cited_reference,
                                                                                filtered_references,filtered_title,filtered_description)
            
            print("CITATION ENDS",time.time() * 1000)
            follow_up_questions = followup_qns.generate_followup_qns(annotated_response)
            #just for testing updated_response = "For queries related to the Filecoin Virtual Machine (FVM), you can connect with the following resources and communities:\nSlack: Join the #fil-builders channel on Slack to engage with other builders and get support from the community [1].\nDeveloper Forum: You can raise questions and start discussions with builders on the FVM Developer Forum [1].\nGitHub: For technical issues and contributions, you can explore the reference FVM implementation and other resources on the GitHub repository [2].\nContact Information: For direct contact, you can reach out to Raul at Protocol Labs via email at raul@protocol.ai [3].\nThese resources should provide you with ample support and information for any queries related to FVM.You can also connect with the marketting team at merketters@protocol.ai"
            if self.validate_presense_pers_info(updated_response):
                masked_response = self.mask_personal_info(updated_response )
                if "error" in masked_response:
                    print("error while trying to mask the pers info")
                    return masked_response
                else:
                    print("masked the personal info")
                    answer = masked_response["masked_message"]
            else:
                print("no personal info")
                answer = updated_response


            #print("*"*70)
            #print(answer)
            #print("*"*70)
            return answer, reorganized_references, follow_up_questions,reorganized_title,reorganized_description
        else:
            return "No response generated."

    def retrieve_context(self, query, chat_history,substack_check):
        full_context = ' '.join(
            [entry['content'] for entry in chat_history if isinstance(entry, dict) and entry['role'] == 'user'])
        if not isinstance(query, str):
            raise TypeError(f"Expected query to be a string, got {type(query)}")
        if not isinstance(full_context, str):
            raise TypeError(f"Expected full_context to be a string, got {type(full_context)}")
        compressor = FlashrankRerank(top_n=self.no_rerank_documents)
        store = self.store
        if substack_check:
            current_date = datetime.now().date()
            if ("recent" in query) or ("trending" in query) or ("latest" in  query):
                days_filter=180
                ret_date= current_date - timedelta(days=days_filter)
            else:
                ret_date=current_date - timedelta(days=365)

            # print("ret date----------->",ret_date)
            # print(f"ret_date before conversion: {ret_date}, type: {type(ret_date)}")
            if isinstance(ret_date, datetime):
                ret_date = ret_date.replace(hour=0, minute=0, second=0, microsecond=0)
            elif isinstance(ret_date, date):
                ret_date = datetime.combine(ret_date, datetime.min.time())
            else:
                raise ValueError("ret_date must be a date or datetime object")
            # print(f"ret_date after conversion: {ret_date}, type: {type(ret_date)}")

            # Convert ret_date to neo4j.time.DateTime
            neo4j_ret_date = Neo4jDateTime(ret_date.year, ret_date.month, ret_date.day, ret_date.hour, ret_date.minute,
                                    ret_date.second, ret_date.microsecond)
            filter={"date":{"gt":neo4j_ret_date}}
        else:
            filter = None

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=store.as_retriever(search_kwargs={"k": self.no_matches}, return_source_documents=True,filter = filter)
        )
        return compression_retriever.invoke(query, context=full_context)

    def format_context(self, context_docs):
        context = "\n".join([self.format_document(doc, index + 1, self.combine_docs_chain.document_prompt) for index, doc in
                             enumerate(context_docs)])
        return context
    #
    # def get_answer(self, query, chat_history):
    #     self.chat_history.append({"role": "user", "content": query})
    #     question = self.create_standalone_question(query, self.chat_history)
    #     if not isinstance(question, str):
    #         raise TypeError(f"Expected question to be a string, got {type(question)}")
    #     context_docs = self.retrieve_context(question, self.chat_history)
    #     context = self.format_context(context_docs)
    #
    #     # Debug Statements
    #     print("Standalone Question:", question)
    #     print("Chat History:", self.chat_history)
    #     print("Context Documents:", context_docs)
    #     print("Formatted Context:", context)
    #
    #     context_question = {
    #         "context": context,
    #         "chat_history": self.chat_history,
    #         "question": question,
    #     }
    #     answer, references, follow_up_questions = self.generate_response(context_question, context_docs)
    #     self.chat_history.append({"role": "assistant", "content": answer})
    #     return answer, references, follow_up_questions

    def extract_cited_references(self, text):
        pattern = r'\[\d+\]'
        cited_refs = re.findall(pattern, text)
        cited_refs = [ref.strip('[]') for ref in cited_refs]
        return cited_refs

    def parse_reference_list(self, reference_list_str):
        references = []
        lines = reference_list_str.strip().split('\n')
        for line in lines:
            if line.strip():
                parts = line.split('. ', 1)
                if len(parts) == 2:
                    reference_number, source = parts
                    references.append((reference_number.strip(), source.strip()))
        return references

    def filter_references_to_keep(self, reference_list_str, cited_references):
        reference_list = self.parse_reference_list(reference_list_str)
        filtered_references = []
        seen_refs = set()
        for ref_number in cited_references:
            for ref in reference_list:
                if ref[0] == ref_number and ref_number not in seen_refs:
                    filtered_references.append(ref)
                    seen_refs.add(ref_number)
                    break
        return filtered_references
    
    def filter_titles_to_keep(self, reference_list_str, cited_references):
        reference_list = self.parse_reference_list(reference_list_str)
        filtered_references = []
        seen_refs = set()
        for ref_number in cited_references:
            for ref in reference_list:
                if ref[0] == ref_number and ref_number not in seen_refs:
                    filtered_references.append(ref)
                    seen_refs.add(ref_number)
                    break
        return filtered_references

    def renumber_references(self, response, cited_references, filtered_references,filtered_title,filtered_description):
        old_to_new_mapping = {old: str(new + 1) for new, (old, _) in enumerate(filtered_references)}
        pattern = re.compile(r'\[(\d+)\]')

        def replace_citation(match):
            old_number = match.group(1)
            new_number = old_to_new_mapping.get(old_number, old_number)
            return f'[{new_number}]'

        updated_response = pattern.sub(replace_citation, response)
        reorganized_references = [(str(i + 1), ref[1]) for i, ref in enumerate(filtered_references)]
        reorganized_title = [(str(i + 1), ref[1]) for i, ref in enumerate(filtered_title)]
        reorganized_description = [(str(i + 1), ref[1]) for i, ref in enumerate(filtered_description)]
        return updated_response, reorganized_references, reorganized_title,reorganized_description


# Example usage
# if __name__ == "__main__":
#     chatbot = Chatbot()
#     chat_history = [{"role": "user", "content": "what is protocol labs"}]
#     query = "who are its employees"
#     response, sources, followup_qns = chatbot.get_answer(query, chat_history)
#     print(response)
#     print("*" * 120)
#     print(sources)
#     print("*" * 120)
#     print(followup_qns)
#     print(type(followup_qns))