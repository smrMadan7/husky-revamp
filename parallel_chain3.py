import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from openai import OpenAI
import copy
from database import Storage
from text_citation import StuffDocumentsWithIndexChain
import time
import functools


load_dotenv()

client = OpenAI()




def time_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # End time
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds")
        return result
    return wrapper



def time_async_function(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time
        result = await func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # End time
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds")
        return result
    return wrapper
'''Your role is to provide detailed information based on the following sources. 
        Please provide information on the topic but do not include any contact details such as phone numbers, emails, LinkedIn profiles, or Twitter handles.
        When referencing the documents, add a citation right after. Use "[SOURCE_NUMBER](source_url)" for the citation (e.g. "The Space Needle is in Seattle [1](https://www.space.com)[2](https://www.astro_space.com).")
        Please provide 3 followup questions under heading Follow-up Questions without numbering them according to the context.'''

class Chatbot:
    def __init__(self):
        # Initialize ChatOpenAI with streaming enabled
        self.llm_chain = LLMChain(
            llm=ChatOpenAI(model=os.getenv('retrieval_model'), max_tokens=int(os.getenv('max_tokens')),
                           temperature=float(os.getenv("temperature")),streaming=True),
            prompt=self.combine_doc_prompt
        )

    combine_doc_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""
         Please provide in-depth and well-organized information on the topic based on sources present in the context.
        Cite your sources directly after the referenced information using this format:
        Use "[SOURCE_NUMBER](source_url)" for the citation (e.g. "The Space Needle is in Seattle [1](https://www.space.com)[2](https://www.astro_space.com)
        Ensure that your response is informative and helpful, summarizing key points while linking the information back to its original source.
 

After your response, generate follow-up questions based on the information you've provided in the response under heading Follow-up Questions, rather than the initial context.
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

    # Cache for storing frequent queries (TTL of 1 hour)
    # response_cache = TTLCache(maxsize=100, ttl=3600)

    def __init__(self, chat_history=None):
        self.temperature = float(os.getenv("temperature"))
        self.model = os.getenv("model")
        self.retrieval_model = os.getenv("retrieval_model")
        self.streaming = bool(os.getenv("streaming"))
        self.no_matches = int(os.getenv("no_matches"))
        self.no_rerank_documents = int(os.getenv("no_rerank_documents"))
        self.max_tokens = int(os.getenv("max_tokens"))
        self.store = Storage().connect_storage()
        self.store.index_name = 'PL_Twitter_Large'
        self.store.node_label = 'Twitter_Chunk'
        self.chat_history = chat_history if chat_history is not None else []
        self.retrieval_chain = self.create_retrieval_chain()

    def encapsulate_references(self,text):
        pattern = r'(\[\d+\]\(http[^\)]+\))'
        modified_text = re.sub(pattern, r'[\1]', text)
        return modified_text


    async def stream_answer(self, query, chat_history):
        """Handles streaming of tokens for the response."""
        chat_history.append({"role": "user", "content": query})
        question = self.create_standalone_question(query, chat_history)

        # Retrieve the context
        context_docs = await self.retrieve_documents_in_parallel(question, chat_history)
        context_docs = [doc for doc in await context_docs if
                        'source' in doc.metadata]  # Ensure 'source' in metadata

        # Prepare context and question
        context_question = {
            "context": context_docs,
            "chat_history": chat_history[-2:],
            "question": question,
        }

        # Start streaming tokens for the answer
        async for token in self.generate_streamed_response(context_question):
            print(token, end='', flush=True)  # Print tokens as they stream

    async def generate_streamed_response(self, context_question):
        """Streams the response tokens asynchronously."""
        prompt = self.combine_doc_prompt.format(**context_question)

        # Get the streaming response from the LLMChain
        async for token in self.llm_chain.run_stream(prompt):
            yield token  # Yield each token as it's generated

    @time_async_function
    async def get_answer(self, query, chat_history):
        # if query in self.response_cache:
        #     return self.response_cache[query]
        chat_history.append({"role": "user", "content": query})
        question = self.create_standalone_question(query, chat_history)
        context_docs = await self.retrieve_documents_in_parallel(question, chat_history)
        print(context_docs)

        context_docs = [doc for doc in await context_docs if 'source' in doc.metadata]  # Await the coroutine



        # context_docs = await self.retrieve_documents_in_parallel(question, chat_history)
        #context_docs = [doc for doc in context_docs if 'source' in doc.metadata]
        context_question = {
            "context": context_docs,
            "chat_history": chat_history[-2:],
            "question": question,
        }
        print(context_question)
        answer, references, follow_up_questions,filtered_title, filtered_description= await self.generate_response(context_question, context_docs)
        # self.response_cache[query] = (answer, references, follow_up_questions)
        answer=self.encapsulate_references(answer)
        chat_history.append({"role": "assistant", "content": answer})
        sources = self.format_source(references, filtered_title, filtered_description)
        return answer,references,follow_up_questions,sources


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
        print("RETURN SOURCE FORMATTING",time.time() * 1000)
        return final_source


    def create_standalone_question(self, question, chat_history):
         if not hasattr(self, 'contextualize_q_chain'):
            model = ChatOpenAI()  # Initialize model once
            contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just repharse it if needed and otherwise return it as is."""
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "question - {question}, chat history - {chat_history}"),
                ]
            )
            self.contextualize_q_chain = contextualize_q_prompt | model | StrOutputParser()
         response = self.contextualize_q_chain.invoke({"question": question, "chat_history": chat_history})
         return response.get('text', question) if isinstance(response, dict) else question

    def create_retrieval_chain(self):
        compressor = FlashrankRerank(model='ms-marco-TinyBERT-L-2-v2',top_n=self.no_rerank_documents)
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

    def format_document(self, doc, index, prompt):
        base_info = {"page_content": doc.page_content, "index": index, "source": doc.metadata['source']}
        document_info = {k: base_info[k] for k in prompt.input_variables}
        return prompt.format(**document_info)



    @time_async_function
    async def retrieve_documents_in_parallel(self, query, chat_history):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(ThreadPoolExecutor(), self.retrieve_context, query, chat_history)
        return result

    def merge_page_contents(self,data):
        if 'context' in data:
            merged_content = ""
            for doc in data['context']:
                # Access 'page_content' as an attribute, not a key
                if hasattr(doc, 'page_content'):
                    merged_content += getattr(doc, 'page_content') + "\n\n"
                else:
                    print(f"'page_content' not found in document: {doc}")
            return merged_content.strip()
        else:
            return "No context available."
    @time_async_function
    async def retrieve_context(self, query, chat_history):
        self.compressor = FlashrankRerank(model='ms-marco-TinyBERT-L-2-v2', top_n=self.no_rerank_documents)
        self.base_retriever = self.store.as_retriever(search_kwargs={"k": self.no_matches}, return_source_documents=True)

        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.base_retriever
        )
        full_context = ' '.join(
            entry['content'] for entry in chat_history
            if isinstance(entry, dict) and entry['role'] == 'user'
        )
        documents = await asyncio.to_thread(self.compression_retriever.invoke, query, context=full_context)
        return documents

    # def retrieve_context(self, query, chat_history):
    #     self.compressor = FlashrankRerank(model='ms-marco-TinyBERT-L-2-v2',top_n=self.no_rerank_documents)
    #     self.compression_retriever = ContextualCompressionRetriever(
    #         base_compressor=self.compressor,
    #         base_retriever=self.store.as_retriever(search_kwargs={"k": self.no_matches}, return_source_documents=True)
    #     )
    #     full_context = ' '.join(
    #         entry['content'] for entry in chat_history
    #         if isinstance(entry, dict) and entry['role'] == 'user'
    #     )
    #     return self.compression_retriever.invoke(query, context=full_context)

    def validate_presense_pers_info(self, message):
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        phone_pattern = r'(?<!\S)(\+?\d{1,4}[\s-]?)?(?!0+\s*,?$)(\d{10,14})(?!\S)'
        emails = re.findall(email_pattern, message)
        phone_numbers = re.findall(phone_pattern, message)
        phone_numbers = [match[1] for match in phone_numbers]
        if len(emails) > 0 or len(phone_numbers) > 0:
            return True
        else:
            return False

    # validate_presense_pers_info(message)
    def mask_personal_info(self, message):
        print(" MASKING  BEGINS", time.time() * 1000)

        try:
            model = ChatOpenAI(temperature=0)

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
            print(" MASKING ENDS", time.time() * 1000)
            return {"masked_message": masked_message}
        except Exception as e:
            return {"message": "unable to find or mask the personal information", "error": str(e)}

    def format_context(self, context_docs):
        context = "\n".join([self.format_document(doc, index + 1, self.combine_docs_chain.document_prompt) for index, doc in enumerate(context_docs)])
        return context

    def extract_followup_qns(self,text):
        questions_list = [line.strip() for line in text.split('\n') if line.strip()]
        questions_dict = {
            "questions": questions_list
        }
        print(questions_dict)
        return questions_dict


    @time_async_function
    async def generate_response(self, context_question, context_docs):
        # Generate response from combine_docs_chain
        llm_response = await self.combine_docs_chain.acall(
            {
                "context": context_question["context"],
                "chat_history": context_question["chat_history"],
                "question": context_question["question"],
                "input_documents": context_docs
            }
        )

        filt_response=llm_response['output_text'].split("Follow-up Questions")
        filt_response[1]=re.sub("-","",filt_response[1])
        follow_up_questions=self.extract_followup_qns(filt_response[1])
        # text_response=filt_response[0].split("References")

        if llm_response and filt_response[0]:
            references = "\n\nReferences:\n"
            title = "\n\ntitle:\n"
            description = "\n\ndescription: \n"

            for i, doc in enumerate(context_docs):
                print(doc.metadata['source'])
                references += f"{i + 1}. {doc.metadata['source']}\n"

                if 'title' in doc.metadata:
                    title += f"{i + 1}. {doc.metadata['title']}\n"
                else:
                    title += f"{i + 1}. {None}\n"
                if 'description' in doc.metadata:
                    description += f"{i + 1}. {doc.metadata['description']}\n"
                else:
                    description += f"{i + 1}. {None}\n"

            annotated_response = filt_response[0]
            references = "\n\nReferences:\n" + "\n".join(
                f"{i + 1}. {doc.metadata['source']}" for i, doc in enumerate(context_docs)
            )
            print(references)
            cited_reference = self.extract_cited_references(annotated_response)
            print(f'cited_reference{cited_reference}')
            filtered_references = self.filter_references_to_keep(references, cited_reference)
            print(f"filtered_reference{filtered_references}")
            filtered_title = self.filter_titles_to_keep(title, cited_reference)
            filtered_description = self.filter_titles_to_keep(description, cited_reference)
            updated_response, reorganized_references, reorganized_title, reorganized_description = self.renumber_references(
                annotated_response, cited_reference,filtered_references, filtered_title, filtered_description)
            return updated_response, reorganized_references, follow_up_questions, reorganized_title, reorganized_description
        return "No response generated.", None, None

    def extract_cited_references(self, text):
        pattern = r'\[\d+\]'
        cited_refs = re.findall(pattern, text)
        return [ref[1:-1] for ref in cited_refs]

    def parse_reference_list(self, reference_list_str):
        return [
            (parts[0].strip(), parts[1].strip())
            for line in reference_list_str.strip().split('\n')
            if line.strip() and len((parts := line.split('. ', 1))) == 2
        ]

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


    def filter_references_to_keep(self, reference_list_str, cited_references):
        reference_list = self.parse_reference_list(reference_list_str)
        cited_ref_set = set(cited_references)
        filtered_references = {
            ref[0]: ref for ref in reference_list if ref[0] in cited_ref_set
        }
        return list(filtered_references.values())

    def renumber_references(self, response, cited_references, filtered_references, filtered_title,
                            filtered_description):
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
        return updated_response, reorganized_references, reorganized_title, reorganized_description


if __name__ == "__main__":
    async def main():
        chatbot = Chatbot()
        chat_history = [{"role": "user", "content": ""}]
        query = "Provide me the updates in filecoin core devs meeting 64"
        response,references,followup_qns,sources = await chatbot.get_answer(query, chat_history)
        print(response)
        print(references)
        print(sources)

    asyncio.run(main())
