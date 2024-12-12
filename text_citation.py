from langchain.chains import StuffDocumentsChain


class StuffDocumentsWithIndexChain(StuffDocumentsChain):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def format_document(self,doc,index,prompt):
        if "source" in doc.metadata:
            base_info = {"page_content": doc.page_content, "index": index, "source": doc.metadata['source']}
        else: 
            doc.metadata['source'] = "unavaialable"
            base_info = {"page_content": doc.page_content, "index": index, "source": doc.metadata['source']}
        missing_metadata = set(prompt.input_variables).difference(base_info)
        if len(missing_metadata) > 0:
            raise ValueError(f"Missing metadata: {list(missing_metadata)}.")
        document_info = {k: base_info[k] for k in prompt.input_variables}
        return prompt.format(**document_info)

    def format_tweet(self,doc, index, prompt):
        if "source" not in doc.metadata:
            doc.metadata['source'] = None
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

    def _get_inputs(self, docs, **kwargs):
        doc_strings = []
        for i, doc in enumerate(docs, 1):
            if "date" in doc.metadata:
                doc_strings.append(self.format_tweet(doc, i, self.document_prompt))
            else:
                doc_strings.append(self.format_document(doc, i, self.document_prompt))

        inputs = {k: v for k, v in kwargs.items() if k in self.llm_chain.prompt.input_variables}
        inputs[self.document_variable_name] = self.document_separator.join(doc_strings)
        return inputs


