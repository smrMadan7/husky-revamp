import os
from chain3 import Chatbot
from datetime import datetime, timedelta
import followup_qns
from langchain.schema import Document
import re
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from text_citation import StuffDocumentsWithIndexChain
load_dotenv()
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
# from rerouter import RerouteData
import copy
import time


class Tweets(Chatbot):
    combine_tweet_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=""" Your role is to provide detailed information based on the following sources. 
    When referencing the documents, add a citation right after. Use "[SOURCE_NUMBER](source_url)" for the citation (e.g. "The Space Needle is in Seattle [1](https://www.space.com)[2](https://www.astro_space.com).")

    Sources:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}"""
    )

    combine_tweet_chain = StuffDocumentsWithIndexChain(
        llm_chain=LLMChain(
            llm=ChatOpenAI(model=os.getenv('retrieval_model'), max_tokens=int(os.getenv('max_tokens')),
                           temperature=float(os.getenv("temperature"))),
            prompt=combine_tweet_prompt,
        ),
        document_prompt=PromptTemplate(
            input_variables=["index", "source", "page_content", "date", "author_name", "mentioned_urls"],
            template="[{index}] {source} (by {author_name} on {date}):\n{page_content}\nMentioned URLs: {mentioned_urls}",
        ),
        document_variable_name="context",
    )

    def __init__(self, chat_history=None):
        self.twitter_no_matches = int(os.getenv("twitter_no_matches"))
        self.twitter_no_rerank_documents = int(os.getenv("twitter_no_rerank_documents"))
        super().__init__(chat_history=chat_history)
        self.store.index_name='PL_Twitter_Large'
        self.store.node_label='Tweet_chunk'
        self.summarizer=os.getenv('summarizing_model')
        self.cut_off_days = 120 # Set custom attribute for Tweets class
        # print(self.cut_off_days,self.store.index_name,self.store.node_label)


    def format_document(self,doc, index, prompt):
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

    def filter_documents_by_date(self, docs, days):
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_docs = [doc for doc in docs if datetime.strptime(doc.metadata['date'], '%Y-%m-%d') > cutoff_date]
        return filtered_docs

    def determine_cut_off_days(self,query):
        # Check if the word 'recent' is present in the query (case-insensitive)
        if 'recent'  in query.lower() or 'latest'  in query.lower():
            days = 60
        else:
            days = 120

        return days

    def summarize_text(self,text):
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





    # def extract_entities(self,answer,references):
    #     extracted_sources = [item[1] for item in references]
    #     reference="\n".join(extracted_sources)
    #     text=" ".join([answer,reference])
    #     openai_llm = ChatOpenAI(model="gpt-3.5-turbo")
    #     prompt_template = PromptTemplate(
    #         input_variables=["text"],
    #         template="""
    #         Extract the names of people and organizations from the following text and categorize them as 'Person' or 'Organization':
    #
    #         Text: "{text}"
    #
    #         Output the results in a JSON format with 'person' and 'organization' lists.
    #         """
    #     )
    #     chain = LLMChain(llm=openai_llm, prompt=prompt_template)
    #     response = chain.run({"text": text})
    #     entities = eval(response)
    #     print("Entities extracted are --------------->", entities)
    #     return entities

    def filter_info(self,additional_info):
        # Define the threshold
        threshold = 0.6

        # Filter the list of lists
        filtered_additional_info = []

        for sublist in additional_info:
            filtered_sublist = []
            for i in range(0, len(sublist), 2):
                doc = sublist[i]
                score = sublist[i][1]
                if score >= threshold:
                    filtered_sublist.extend([doc])
            if filtered_sublist:
                filtered_additional_info.append(filtered_sublist)
        print(filtered_additional_info)
        return filtered_additional_info



    def get_answer(self, query, chat_history, days_filter=None):

        cut_off_days=self.determine_cut_off_days(query)
        # print(cut_off_days,type(cut_off_days))
        self.chat_history.append({"role": "user", "content": query})
        question = self.create_standalone_question(query, self.chat_history)
        if not isinstance(question, str):
            raise TypeError(f"Expected question to be a string, got {type(question)}")
        context_docs = self.retrieve_context(question, self.chat_history)
        if cut_off_days is not None:
            context_docs = self.filter_documents_by_date(context_docs, cut_off_days)
        context_question = {
            "context": context_docs,
            "chat_history": self.chat_history,
            "question": question,
        }
        # print("Context Docs---------->",context_docs)
        # print("Context_question-------------->",context_question)
        # print("*"*120)
        answer, references,sources, follow_up_questions = self.generate_response(context_question, context_docs)
        # print("Response: ------------------->",answer)
        self.chat_history.append({"role": "assistant", "content": answer})
        
        """reroute_instance = RerouteData()
        entities=reroute_instance.extract_entities(answer,references)
        self.store.index_name="PL_Twitter_Large"#PL_Twitter_Large
        self.store.node_label="tweet_chunk"#"Twitter_Chunk"#tweet_chunk
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
            summary_enumerated = None"""
        return answer, sources, follow_up_questions,references#,summary_enumerated
    def generate_response(self, context_question, context_docs):
        response = self.combine_tweet_chain.invoke(
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
                #updated_citation = f"{source_id} (Tweet from {tweet_date})"
                #annotated_response = annotated_response.replace(source_id, updated_citation)
            cited_reference = self.extract_cited_references(annotated_response)
            filtered_references = self.filter_references_to_keep(references, cited_reference)
            # Temporary hotfix for modular function renumber_references
            title=""
            desc=""
            updated_response, reorganized_references,title,desc = self.renumber_references(annotated_response, cited_reference,
                                                                                filtered_references,title,desc)

            follow_up_questions = followup_qns.generate_followup_qns(annotated_response)
            print(references)
            sources = self.format_source(reorganized_references)
            return updated_response,reorganized_references, sources, follow_up_questions
            #return updated_response, reorganized_references, follow_up_questions
        else:
            return "No response generated."
    def format_source(self,references):
        source_dict = {
            "id": "",
            "link": "",
            "title": "",
            "handle": "",
            "date":""
        }

        final_source = []
        print(references)
        for i in range(0,int(len(references))):
            source_dict["id"] = references[i][0]
            user_data = references[i][1]
            handle,url,by_word, on_word = self.extract_user_data( user_data)
            source_dict["link"] = url
            source_dict["title"] = by_word
            source_dict["handle"] = handle
            source_dict["date"]   = on_word

            #source_dict["description"] = filtered_description[i][1]
            final_source = final_source + [copy.deepcopy(source_dict)]
        print(final_source)
        print("final_source")
        return final_source
    
    def extract_user_data(self, input):
        import re
        # Regex patterns to extract the required parts
        handle_regex = r'(@\w+)'
        by_regex = r'\bby ([\w\s]+?)\s+on'
        on_regex = r'\bon (\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})\b'
        url_regex = r'https?://\S+'

        # Find the handle using the regex
        handle_match = re.search(handle_regex, input)
        handle = handle_match.group(1) if handle_match else None

        # Find the words after "by" using the regex
        by_match = re.search(by_regex, input)
        by_word = by_match.group(1) if by_match else None

        # Find the date after "on" using the regex
        on_match = re.search(on_regex, input)
        on_word = on_match.group(1) if on_match else None

        # Find the URL using the regex
        url_match = re.search(url_regex, input)
        url = url_match.group(0) if url_match else None

        print(f"Handle: {handle}")
        print(f"By: {by_word}")
        print(f"On: {on_word}")
        print(f"URL: {url}")

        return handle,url,by_word, on_word

    # def invoke_retriever(self, query: str) -> str:
    #     self.store.index_name="PL_Twitter_Large"
    #     self.store.node_label="Twitter_Chunk"
    #     add_additional_responses=self.store.similarity_search(query,k=5)
    #     return add_additional_responses
    #
    # def process_queries_concurrently(self, queries: List[str]) -> Tuple[List[str], List[str]]:
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = {executor.submit(self.invoke_retriever, query): query for query in queries}
    #         paired_results = []
    #         for future in concurrent.futures.as_completed(futures):
    #             query = futures[future]
    #             try:
    #                 result = future.result()
    #                 paired_results.append((query, result))  # Append the query and its result as a tuple
    #             except Exception as exc:
    #                 paired_results.append(
    #                     (query, f"Error processing {query}: {exc}"))  # Include the query in case of an error
    #
    #     queries_list, results_list = zip(*paired_results) if paired_results else ([], [])
    #     return list(queries_list), list(results_list)

    # def process_queries_concurrently(self, queries: List[str]) -> List[str]:
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = {executor.submit(self.invoke_retriever, query): query for query in queries}
    #         results = []
    #         for future in concurrent.futures.as_completed(futures):
    #             query = futures[future]
    #             try:
    #                 result = future.result()
    #                 results.append(result)
    #             except Exception as exc:
    #                 results.append(f"Error processing {query}: {exc}")
    #     return results


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

    def extract_date_from_context(self,test_str):
        match_str = re.search(r'\d{4}-\d{2}-\d{2}', test_str)
        if not match_str:
            return "None"
        res = datetime.strptime(match_str.group(), '%Y-%m-%d').date()
        return res

    def filter_docs(self,doc,current_date):
        try:
            doc_date = self.extract_date_from_context(doc)
        except (ValueError, KeyError):
            return False
        if doc_date == "None":
            return False
        if (current_date - doc_date).days > self.cut_off_days:
            return False

        return True

    def retrieve_context(self, query, chat_history):
        full_context = ' '.join(
            [entry['content'] for entry in chat_history if isinstance(entry, dict) and entry['role'] == 'user'])
        if not isinstance(query, str):
            raise TypeError(f"Expected query to be a string, got {type(query)}")
        if not isinstance(full_context, str):
            raise TypeError(f"Expected full_context to be a string, got {type(full_context)}")
        current_date = datetime.now().date()
        retriever = self.store.as_retriever(search_kwargs={"k": self.twitter_no_matches}, return_source_documents=True)
        documents = retriever.invoke(query)
        filtered_documents = [doc.page_content for doc in documents if self.filter_docs(doc.page_content,current_date)]
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
                        # print("Parsed Documents -------------->",parsed_documents)
        return parsed_documents

    #Please answer the question with citation to the paragraphs. For every sentence you write, cite the source name and reference number provided.
                       # Provide answers in a structured way with an introduction, main content with sub-sections, and a conclusion.

    def create_retrieval_chain(self):
        compressor = FlashrankRerank(top_n=self.twitter_no_rerank_documents)
        store = self.store
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=store.as_retriever(search_kwargs={"k": self.twitter_no_matches}, return_source_documents=True)
        )
        chat_model = ChatOpenAI(temperature=self.temperature, model=self.retrieval_model, max_tokens=self.max_tokens,
                                streaming=self.streaming)

        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="""Your task is to retrieve and provide detailed information based on the following criteria:
                        Your task is to reframe questions to include all relevant interchangeable terms if they contain any one or more of the keywords: summits, conferences, hackathons, workshops, or events. Treat these terms as interchangeable. If the question contains one or more of these keywords, reframe it to include all five terms.

                        ### Example:

                        Original Question: "News about recent workshops."
                        Reframed Question: "News about recent workshops, events, conferences, summits, hackathons."
                        
                        ### Steps to Follow:
                        
                        1. **Detect Keywords**:
                           - Check if the question contains any of the keywords: summits, conferences, hackathons, workshops, or events.
                        
                        2. **Reframe the Question**:
                           - If the question contains one or more of these keywords, reframe it to include all five terms: summits, conferences, hackathons, workshops, and events.
                        
                        3. **Provide the Reframed Question**:
                           - Output the reframed question.

                    ### Example Prompts:

       
                     
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




#
if __name__ == "__main__":
    tbot = Tweets()
    question = "What are recent updates on filecoin ecosystem"
    chat_history = [
        {"role": "user", "content": ""},
          ]
    answer, references, follow_up_questions,augmented_data = tbot.get_answer(question, chat_history)
    print("Answer:", answer)
    print("References:", references)
    print("Follow-up Questions:", follow_up_questions)
    print("Additional_Info------------------->",augmented_data)



