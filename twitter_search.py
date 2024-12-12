import os
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import StuffDocumentsChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import  StrOutputParser
from langchain.retrievers.document_compressors import FlashrankRerank
from openai import OpenAI
import re
import followup_qns
from dotenv import load_dotenv
from database import Storage
load_dotenv()
from datetime import datetime, timedelta
import datefinder
from dateutil.parser import parse as date_parse
from langchain.schema import Document
import json

client = OpenAI()

storage_instance = Storage()

def format_document(doc, index, prompt):
    base_info = {
        "page_content": doc.page_content,
        "index": index,
        "source": doc.metadata['source'],
        "date": doc.metadata.get('date', 'Unknown date'),
        "url": doc.metadata.get('url', 'No URL'),
        "author_name": doc.metadata.get('author_name', 'Unknown author'),
        "mentioned_urls": ", ".join(doc.metadata.get('mentioned_urls', []))
    }
    missing_metadata = set(prompt.input_variables).difference(base_info)
    if len(missing_metadata) > 0:
        raise ValueError(f"Missing metadata: {list(missing_metadata)}.")
    document_info = {k: base_info[k] for k in prompt.input_variables}
    return prompt.format(**document_info)



class StuffDocumentsWithIndexChain(StuffDocumentsChain):
    def _get_inputs(self, docs, **kwargs):
        doc_strings = [
            format_document(doc, i, self.document_prompt)
            for i, doc in enumerate(docs, 1)
        ]
        inputs = {k: v for k, v in kwargs.items() if k in self.llm_chain.prompt.input_variables}
        inputs[self.document_variable_name] = self.document_separator.join(doc_strings)
        return inputs


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
        input_variables=["index", "source", "page_content", "date", "author_name", "mentioned_urls"],
        template="[{index}] {source} (by {author_name} on {date}):\n{page_content}\nMentioned URLs: {mentioned_urls}",
    ),
    document_variable_name="context",
)



def extract_date_from_context(test_str):
    match_str = re.search(r'\d{4}-\d{2}-\d{2}', test_str)
    if not match_str:
        return "None"
    res = datetime.strptime(match_str.group(), '%Y-%m-%d').date()
    return res


class Chatbot:
    def __init__(self, chat_history=None,cut_off_days=30):
        self.temperature = float(os.getenv("temperature"))
        self.model = os.getenv("model")
        self.retrieval_model = os.getenv("retrieval_model")
        self.streaming = bool(os.getenv("streaming"))
        self.no_matches = int(os.getenv("no_matches"))
        self.store = storage_instance.connect_storage()  # Ensure connection is reused
        self.no_rerank_documents = int(os.getenv("no_rerank_documents"))
        self.max_tokens = int(os.getenv("max_tokens"))
        self.retrieval_chain = self.create_retrieval_chain()
        self.chat_history = chat_history if chat_history is not None else []
        self.cutoff_days=cut_off_days


    def format_document(doc, index, prompt):
        base_info = {
            "page_content": doc.page_content,
            "index": index,
            "source": doc.metadata['source'],
            "date": doc.metadata.get('date', 'Unknown date'),
            "url": doc.metadata.get('url', 'No URL'),
            "author_name": doc.metadata.get('author_name', 'Unknown author')
        }
        missing_metadata = set(prompt.input_variables).difference(base_info)
        if len(missing_metadata) > 0:
            raise ValueError(f"Missing metadata: {list(missing_metadata)}.")
        document_info = {k: base_info[k] for k in prompt.input_variables}
        return prompt.format(**document_info)

    def filter_documents_by_date(self,docs, days):
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_docs = [doc for doc in docs if datetime.strptime(doc.metadata['date'], '%Y-%m-%d') > cutoff_date]
        return filtered_docs



    def get_answer(self, query, chat_history, days_filter=None):
        self.chat_history.append({"role": "user", "content": query})
        question = self.create_standalone_question(query, self.chat_history)
        if not isinstance(question, str):
            raise TypeError(f"Expected question to be a string, got {type(question)}")

        context_docs = self.retrieve_context(question, self.chat_history)

        if days_filter is not None:
            context_docs = self.filter_documents_by_date(context_docs, days_filter)

        context_question = {
            "context": context_docs,
            "chat_history": self.chat_history,
            "question": question,
        }
        answer, references, follow_up_questions = self.generate_response(context_question, context_docs)
        self.chat_history.append({"role": "assistant", "content": answer})
        return answer, references, follow_up_questions

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

    def generate_response(self, context_question, context_docs):
        response = combine_docs_chain.invoke(
            {
                "context": context_question["context"],
                "chat_history": context_question["chat_history"],
                "question": context_question["question"],
                "input_documents": context_docs
            }
        )
        if response and response["output_text"]:
            annotated_response = response["output_text"]
            references = "\n\nReferences:\n"
            for i, doc in enumerate(context_docs):
                references += f"{i + 1}. {doc.metadata['source']} by {doc.metadata['author_name']} on {doc.metadata['date']}. URL: {doc.metadata['url']}\n"

            for i, doc in enumerate(context_docs):
                source_id = f"[{i + 1}]"
                tweet_date = doc.metadata.get('date', 'Unknown date')
                updated_citation = f"{source_id} (Tweet from {tweet_date})"
                annotated_response = annotated_response.replace(source_id, updated_citation)

            cited_reference = self.extract_cited_references(annotated_response)
            filtered_references = self.filter_references_to_keep(references, cited_reference)
            updated_response, reorganized_references = self.renumber_references(annotated_response, cited_reference,
                                                                                filtered_references)

            follow_up_questions = followup_qns.generate_followup_qns(annotated_response)
            return updated_response, reorganized_references, follow_up_questions
        else:
            return "No response generated."


    def extract_metadata(self,page_content):
        try:

            text_pattern = r'\{text: (.*?) author_name:'
            match = re.search(text_pattern, page_content, re.DOTALL)

            if not match:
                raise ValueError("Page content format is incorrect")

            text_content = match.group(1).strip()

            metadata_str = page_content.replace(f"{text_content}", "", 1)

            metadata_pattern = r'(\w+):\s?([^\s\{\[]+|\[.*?\]|null|\d+\})'
            matches = re.findall(metadata_pattern, metadata_str)

            metadata = {key.strip(): value.strip(' ,}') for key, value in matches}

            metadata_final = {
                'source': metadata.get('author_handle', 'Unknown source'),
                'date': metadata.get('date', 'Unknown date'),
                'url': metadata.get('url', 'No URL'),
                'author_name': metadata.get('author_name', 'Unknown author'),
                'lang': metadata.get('lang', 'Unknown lang'),
                'mentioned_urls': metadata.get('mentioned_urls', '[]').strip(
                    '[]').split() if 'mentioned_urls' in metadata else [],
                'is_retweet': metadata.get('is_retweet', 'false').lower() == 'true',
                'media_type': metadata.get('media_type', 'No media'),
                'images_urls': metadata.get('images_urls', 'null').strip(
                    'null').split() if 'images_urls' in metadata else [],
                'num_reply': int(metadata.get('num_reply', 0)),
                'num_retweet': int(metadata.get('num_retweet', 0)),
                'num_like': int(metadata.get('num_like', 0))
            }

            doc = Document(page_content=text_content, metadata=metadata_final)
            return doc
        except Exception as e:
            print(f"Error parsing page_content: {e}")
            return None

    def retrieve_context(self, query, chat_history, metadata_filter="https://x.com"):

        full_context = ' '.join(
            [entry['content'] for entry in chat_history if isinstance(entry, dict) and entry['role'] == 'user'])

        # Validate query and context types
        if not isinstance(query, str):
            raise TypeError(f"Expected query to be a string, got {type(query)}")
        if not isinstance(full_context, str):
            raise TypeError(f"Expected full_context to be a string, got {type(full_context)}")

        # Get the current date
        current_date = datetime.now().date()

        # Retrieve documents using the retriever
        retriever = self.store.as_retriever(search_kwargs={"k": self.no_matches}, return_source_documents=True)
        documents = retriever.invoke(query)

        def filter_docs(doc):

            try:
                doc_date = extract_date_from_context(doc)

            except (ValueError, KeyError):
                return False

            if doc_date == "None":
                return False


            if (current_date - doc_date).days > self.cutoff_days:
                return False


            return True

        filtered_documents = [doc.page_content for doc in documents if filter_docs(doc.page_content)]

        filtered_documents = [
            Document(page_content=doc)
            for doc in filtered_documents
        ]

        if not filtered_documents:
            return []

        parsed_documents = []
        for doc in filtered_documents:
            parsed_doc = self.extract_metadata(doc.page_content)
            if parsed_doc:
                parsed_documents.append(parsed_doc)

        return parsed_documents



    def format_context(self, context_docs):
            context = "\n".join([format_document(doc, index + 1, combine_docs_chain.document_prompt) for index, doc in
                                 enumerate(context_docs)])
            return context


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

    def renumber_references(self, response, cited_references, filtered_references):
        old_to_new_mapping = {old: str(new + 1) for new, (old, _) in enumerate(filtered_references)}
        pattern = re.compile(r'\[(\d+)\]')

        def replace_citation(match):
            old_number = match.group(1)
            new_number = old_to_new_mapping.get(old_number, old_number)
            return f'[{new_number}]'

        updated_response = pattern.sub(replace_citation, response)
        reorganized_references = [(str(i + 1), ref[1]) for i, ref in enumerate(filtered_references)]
        return updated_response, reorganized_references
