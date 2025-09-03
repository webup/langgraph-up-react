import concurrent.futures
import os
import sys
import threading

from config import RAGFLOW, RERANK_MODEL
from llm_server import Rerank_LLM
from ragflow import RAGFlowRetrieval


class KB_Retrieval:
    def __init__(self, similarity_score: float = 0.5, top_k: int = 5, max_workers: int = 4):
        self.rag_client = RAGFlowRetrieval(api_key=RAGFLOW.API_KEY, base_url=RAGFLOW.BASE_URL)
        self.similarity_score = similarity_score
        self.top_k = top_k
        self.max_workers = max_workers
        self.chunk_list = []
        self.chunk_content = []
        self.rerank_client = Rerank_LLM(key=RERANK_MODEL.API_KEY, model_name=RERANK_MODEL.MODEL_NAME, base_url=RERANK_MODEL.BASE_URL)
        self._lock = threading.Lock()  # 线程安全锁

    def retrieve(self, question: list[str]):
        results = self.rag_client.batch_retrieve(question, dataset_ids=[RAGFLOW.DATASET_ID], similarity_threshold=self.similarity_score, top_k=self.top_k)
        for result in results:
            chunks = result.get("chunks", [])
            for chunk in chunks:
                if chunk.get("content") not in self.chunk_list:
                    self.chunk_list.append(chunk.get("content"))
                    self.chunk_content.append(chunk)
        
        # 对多个query的召回结果进行rerank
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
        query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        document_template = "<Document>: {doc}{suffix}"
        query = query_template.format(prefix=prefix, instruction=instruction, query=question[-1])
        texts = [document_template.format(doc=doc, suffix=suffix) for doc in self.chunk_list]
        rank = self.rerank_client.similarity(query, texts)
        # 根据rerank结果对chunk_content进行排序
        self.chunk_content = [x for _, x in sorted(zip(rank, self.chunk_content), key=lambda x: x[0], reverse=True)] 
        context = ""
        for chunk in self.chunk_content[:self.top_k]:
            context += f"<chunk>\n{chunk['content']}\n</chunk>\n"
        return context


if __name__ == "__main__":
    from llm_server import LLM
    llm = LLM()
    rewrite_result = llm.query_rewrite("降转政策")
    query_list = list(rewrite_result.values())[:-1]
    rag = KB_Retrieval()
    context = rag.retrieve(query_list)
    result = llm.chat_completion(query_list[-1], context)
    print(result)
    