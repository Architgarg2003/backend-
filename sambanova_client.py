# import os
# import logging
# from typing import List, Dict, Generator
# from openai import OpenAI
# from langchain_openai import ChatOpenAI
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss

# # Configuration Class
# class Config:
#     SAMBANOVA_BASE_URL = "https://api.sambanova.ai/v1/"
#     DEFAULT_MODEL = "Meta-Llama-3.1-70B-Instruct"
#     MAX_TOKENS = 4096
#     TEMPERATURE = 0.7
#     AVAILABLE_MODELS = [
#         "Meta-Llama-3.1-405B-Instruct",
#         "Meta-Llama-3.1-70B-Instruct",
#         "Meta-Llama-3.1-8B-Instruct",
#         "Meta-Llama-3.2-3B-Instruct"
#     ]

# class RAGAssistant:
#     def __init__(self, sambanova_client):
#         self.logger = logging.getLogger(__name__)
#         self.client = sambanova_client
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.context = ""
#         self.context_chunks = []
#         self.context_embeddings = None
#         self.faiss_index = None

#     def set_context(self, context_text: str, chunk_size: int = 300):
#         self.context = context_text
#         self.context_chunks = []
#         self.context_embeddings = None
#         self.faiss_index = None
#         self._chunk_context(context_text, chunk_size)
#         self._create_embeddings()
#         self.logger.info(f"Context set with {len(self.context_chunks)} chunks")

#     def _chunk_context(self, text: str, chunk_size: int = 300):
#         words = text.split()
#         chunks = []
#         current_chunk = []
        
#         for word in words:
#             current_chunk.append(word)
#             if len(' '.join(current_chunk)) > chunk_size:
#                 chunks.append(' '.join(current_chunk))
#                 current_chunk = []
        
#         if current_chunk:
#             chunks.append(' '.join(current_chunk))
        
#         self.context_chunks = chunks

#     def _create_embeddings(self):
#         try:
#             embeddings = self.embedding_model.encode(self.context_chunks)
#             embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
#             dimension = embeddings.shape[1]
#             self.faiss_index = faiss.IndexFlatL2(dimension)
#             self.faiss_index.add(embeddings)
#             self.context_embeddings = embeddings
#             self.logger.info(f"Created embeddings for {len(self.context_chunks)} chunks")
#         except Exception as e:
#             self.logger.error(f"Embedding creation error: {str(e)}")
#             raise

#     def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
#         if self.faiss_index is None or len(self.context_chunks) == 0:
#             return []
        
#         try:
#             query_embedding = self.embedding_model.encode([query])
#             query_embedding = query_embedding / np.linalg.norm(query_embedding)
#             distances, indices = self.faiss_index.search(query_embedding, top_k)
#             relevant_contexts = [self.context_chunks[idx] for idx in indices[0]]
#             return relevant_contexts
#         except Exception as e:
#             self.logger.error(f"Context retrieval error: {str(e)}")
#             return []

#     # def generate_rag_response(self, query: str) -> str:
#     #     if not self.context:
#     #         return "No context provided. Please set context first."
        
#     #     relevant_contexts = self.retrieve_relevant_context(query)
#     #     context_str = "\n".join(relevant_contexts)
#     #     augmented_prompt = f""" you are a helpful assistant that manages the inventory for my shop the below context is the information you have about the shop inventory.
#     #     Relevant Context:
#     #     {context_str}

#     #     Query: {query}

#     #     Instructions:
#     #     1. You are a helpful agent named AURA (advnaced Unified retail automation bot) that manages the inventory for the shop greet the user politely and provide the information they need.
#     #     2. Answer the query using ONLY the relevant context
#     #     3. If the context lacks sufficient information:
#     #        - Explain what information is missing
#     #        - Indicate inability to provide a complete answer with a polite sorry and offer to help with other queries
#     #     4. Be precise, clear, and directly address the query in a helpful manner and give the most relevant information
#     #     5. Provide a complete answer in a single message
#     #     6. Do not provide any irrelevant information
#     #     """
        
#     #     try:
#     #         response = self.client.generate_chat_completion(
#     #             prompt=augmented_prompt, 
#     #             temperature=0.3
#     #         )
#     #         return response
#     #     except Exception as e:
#     #         self.logger.error(f"RAG Response Generation Error: {str(e)}")
#     #         return f"Error generating response: {str(e)}"



#     def generate_rag_response(self, query: str) -> str:
#         if not self.context:
#             return "No context provided. Please set context first."
        
#         # 1. Retrieve relevant context chunks based on query similarity
#         relevant_contexts = self.retrieve_relevant_context(query)
        
#         # 2. Create system message with RAG instructions but without explicitly showing context
#         system_message = """You are AURA (Advanced Unified Retail Automation bot), a helpful assistant that manages inventory for a shop.
#         Follow these guidelines:
#         1. Greet the user politely and provide the information they need
#         2. Use ONLY the retrieved information to answer questions about inventory
#         3. If the information is insufficient, explain what's missing and politely apologize
#         4. Be precise, clear, and directly address the query
#         5. Provide a complete answer in a single message
#         6. Do not provide any irrelevant information
#         """
        
#         # 3. Create a structured conversation with system, context, and user messages
#         messages = [
#             {"role": "system", "content": system_message},
#         ]
        
#         # 4. Add each relevant context as separate "assistant" messages (this is the RAG part)
#         for i, context in enumerate(relevant_contexts):
#             messages.append({
#                 "role": "assistant", 
#                 "content": f"Retrieved information {i+1}: {context}"
#             })
        
#         # 5. Add the user query
#         messages.append({"role": "user", "content": query})
        
#         try:
#             # 6. Use multi-turn conversation method to maintain the full context
#             response = self.client.multi_turn_conversation(messages)
#             return response
#         except Exception as e:
#             self.logger.error(f"RAG Response Generation Error: {str(e)}")
#             return f"Error generating response: {str(e)}"

#     def get_context_stats(self) -> Dict[str, any]:
#         if not self.context:
#             return {"error": "No context available"}
        
#         return {
#             "total_text_length": len(self.context),
#             "chunk_count": len(self.context_chunks),
#             "avg_chunk_size": np.mean([len(chunk) for chunk in self.context_chunks]) if self.context_chunks else 0,
#             "embedding_dimension": self.context_embeddings.shape[1] if self.context_embeddings is not None else 0
#         }

# class SambaNovaClient:
#     def __init__(self, api_key=None, config=Config):
#         self.logger = logging.getLogger(__name__)
#         logging.basicConfig(level=logging.INFO)
#         self.api_key = api_key or os.getenv('SAMBANOVA_API_KEY')
        
#         if not self.api_key:
#             self.logger.error("SambaNova API key is required")
#             raise ValueError("SambaNova API key is required")
        
#         self.config = config
#         self.client = OpenAI(
#             base_url=self.config.SAMBANOVA_BASE_URL,
#             api_key=self.api_key
#         )
        
#         self.langchain_llm = ChatOpenAI(
#             base_url=self.config.SAMBANOVA_BASE_URL, 
#             api_key=self.api_key,
#             streaming=True,
#             model=self.config.DEFAULT_MODEL,
#             temperature=self.config.TEMPERATURE,
#             max_tokens=self.config.MAX_TOKENS
#         )

#         self.rag_assistant = RAGAssistant(self)

#     def set_rag_context(self, context: str, chunk_size: int = 300):
#         self.rag_assistant.set_context(context, chunk_size)

#     def get_rag_response(self, query: str) -> str:
#         return self.rag_assistant.generate_rag_response(query)

#     def generate_chat_completion(
#         self, 
#         prompt: str, 
#         model: str = None, 
#         temperature: float = None
#     ) -> str:
#         try:
#             completion = self.client.chat.completions.create(
#                 model=model or self.config.DEFAULT_MODEL,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=temperature or self.config.TEMPERATURE,
#                 max_tokens=self.config.MAX_TOKENS,
#                 stream=False
#             )
#             return completion.choices[0].message.content
#         except Exception as e:
#             self.logger.error(f"Error in chat completion: {str(e)}")
#             return f"Error generating completion: {str(e)}"

#     def stream_chat_completion(
#         self, 
#         prompt: str, 
#         model: str = None
#     ) -> Generator[str, None, None]:
#         try:
#             completion = self.client.chat.completions.create(
#                 model=model or self.config.DEFAULT_MODEL,
#                 messages=[{"role": "user", "content": prompt}],
#                 stream=True
#             )
#             for chunk in completion:
#                 yield chunk.choices[0].delta.content or ""
#         except Exception as e:
#             self.logger.error(f"Error in streaming completion: {str(e)}")
#             yield f"Error streaming completion: {str(e)}"

#     def langchain_query(
#         self, 
#         prompt: str, 
#         model: str = None
#     ) -> str:
#         try:
#             original_model = self.langchain_llm.model_name
#             if model:
#                 self.langchain_llm.model_name = model

#             response = self.langchain_llm.invoke(prompt)
            
#             if model:
#                 self.langchain_llm.model_name = original_model

#             return response.content
#         except Exception as e:
#             self.logger.error(f"Error with Langchain query: {str(e)}")
#             return f"Error with Langchain query: {str(e)}"

#     def multi_turn_conversation(
#         self, 
#         messages: List[Dict[str, str]]
#     ) -> str:
#         try:
#             completion = self.client.chat.completions.create(
#                 model=self.config.DEFAULT_MODEL,
#                 messages=messages
#             )
#             return completion.choices[0].message.content
#         except Exception as e:
#             self.logger.error(f"Error in multi-turn conversation: {str(e)}")
#             return f"Conversation error: {str(e)}"

#     @classmethod
#     def get_available_models(cls) -> List[str]:
#         return Config.AVAILABLE_MODELS










import os
import logging
from typing import List, Dict, Generator
from openai import OpenAI
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Configuration Class
class Config:
    SAMBANOVA_BASE_URL = "https://api.sambanova.ai/v1/"
    DEFAULT_MODEL = "Meta-Llama-3.1-70B-Instruct"
    MAX_TOKENS = 4096
    TEMPERATURE = 0.7
    AVAILABLE_MODELS = [
        "Meta-Llama-3.1-405B-Instruct",
        "Meta-Llama-3.1-70B-Instruct",
        "Meta-Llama-3.1-8B-Instruct",
        "Meta-Llama-3.2-3B-Instruct"
    ]

class RAGAssistant:
    def __init__(self, sambanova_client):
        self.logger = logging.getLogger(__name__)
        self.client = sambanova_client
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.context = ""
        self.context_chunks = []
        self.context_embeddings = None
        self.faiss_index = None
        self.knowledge_base = {}  # Store full knowledge as structured data

    def set_context(self, context_text: str, chunk_size: int = 300):
        self.context = context_text
        self.context_chunks = []
        self.context_embeddings = None
        self.faiss_index = None
        self._chunk_context(context_text, chunk_size)
        self._create_embeddings()
        self._build_knowledge_base()
        self.logger.info(f"Context set with {len(self.context_chunks)} chunks and knowledge base built")

    def _chunk_context(self, text: str, chunk_size: int = 300):
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        self.context_chunks = chunks

    def _create_embeddings(self):
        try:
            embeddings = self.embedding_model.encode(self.context_chunks)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embeddings)
            self.context_embeddings = embeddings
            self.logger.info(f"Created embeddings for {len(self.context_chunks)} chunks")
        except Exception as e:
            self.logger.error(f"Embedding creation error: {str(e)}")
            raise

    def _build_knowledge_base(self):
        """
        Process the context to build a structured knowledge base that the LLM
        can reference as a whole rather than just retrieving chunks.
        """
        try:
            # Ask the LLM to structure the entire context into a knowledge base
            prompt = f"""
            I'm going to provide you with text about a shop's inventory.
            Please analyze this information and structure it into a comprehensive knowledge base.
            Extract all relevant details such as product names, quantities, prices, categories,
            and any other information that would be useful for inventory management.
            Format the output as a structured JSON.

            Context:
            {self.context}
            """
            
            structured_kb = self.client.generate_chat_completion(
                prompt=prompt,
                temperature=0.1  # Low temperature for more deterministic results
            )
            
            # Store both the raw context and the structured knowledge base
            self.knowledge_base = {
                "raw_context": self.context,
                "structured_data": structured_kb
            }
            
            self.logger.info("Built structured knowledge base from context")
        except Exception as e:
            self.logger.error(f"Knowledge base building error: {str(e)}")
            # Fall back to raw context if knowledge base building fails
            self.knowledge_base = {"raw_context": self.context}

    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        if self.faiss_index is None or len(self.context_chunks) == 0:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            relevant_contexts = [self.context_chunks[idx] for idx in indices[0]]
            return relevant_contexts
        except Exception as e:
            self.logger.error(f"Context retrieval error: {str(e)}")
            return []

    def generate_rag_response(self, query: str) -> str:
        if not self.context:
            return "No context provided. Please set context first."
        
        # First get relevant context chunks for reference
        relevant_contexts = self.retrieve_relevant_context(query)
        
        # Create a hybrid approach that uses both the structured knowledge base
        # and the retrieved relevant context for more comprehensive answers
        system_message = """You are AURA (Advanced Unified Retail Automation bot), a helpful assistant that manages inventory for a shop.
        You have access to a complete knowledge base of the shop's inventory as well as specific relevant information for this query.
        
        Guidelines:
        2. Use BOTH your knowledge base and the retrieved information to provide comprehensive answers
        3. If the information in both sources is insufficient, explain what's missing and politely apologize
        4. Be precise, clear, and directly address the query
        5. Provide a complete answer in a single message
        6. Do not mention that you're using RAG or knowledge bases in your response
        """
        
        # Build a conversation that includes knowledge base and relevant context
        messages = [
            {"role": "system", "content": system_message},
            {"role": "assistant", "content": f"Complete Knowledge Base: {self.knowledge_base.get('structured_data', self.context)}"}
        ]
        
        # Add each relevant context piece as a separate message
        for i, context in enumerate(relevant_contexts):
            messages.append({
                "role": "assistant", 
                "content": f"Additional relevant information {i+1}: {context}"
            })
        
        # Add the user query
        messages.append({"role": "user", "content": query})
        
        try:
            # Use multi-turn conversation to leverage the full context
            response = self.client.multi_turn_conversation(messages)
            return response
        except Exception as e:
            self.logger.error(f"RAG Response Generation Error: {str(e)}")
            return f"Error generating response: {str(e)}"

    def get_context_stats(self) -> Dict[str, any]:
        if not self.context:
            return {"error": "No context available"}
        
        return {
            "total_text_length": len(self.context),
            "chunk_count": len(self.context_chunks),
            "avg_chunk_size": np.mean([len(chunk) for chunk in self.context_chunks]) if self.context_chunks else 0,
            "embedding_dimension": self.context_embeddings.shape[1] if self.context_embeddings is not None else 0,
            "knowledge_base_available": "structured_data" in self.knowledge_base
        }

class SambaNovaClient:
    def __init__(self, api_key=None, config=Config):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.api_key = api_key or os.getenv('SAMBANOVA_API_KEY')
        
        if not self.api_key:
            self.logger.error("SambaNova API key is required")
            raise ValueError("SambaNova API key is required")
        
        self.config = config
        self.client = OpenAI(
            base_url=self.config.SAMBANOVA_BASE_URL,
            api_key=self.api_key
        )
        
        self.langchain_llm = ChatOpenAI(
            base_url=self.config.SAMBANOVA_BASE_URL, 
            api_key=self.api_key,
            streaming=True,
            model=self.config.DEFAULT_MODEL,
            temperature=self.config.TEMPERATURE,
            max_tokens=self.config.MAX_TOKENS
        )

        self.rag_assistant = RAGAssistant(self)

    def set_rag_context(self, context: str, chunk_size: int = 300):
        self.rag_assistant.set_context(context, chunk_size)

    def get_rag_response(self, query: str) -> str:
        return self.rag_assistant.generate_rag_response(query)

    def generate_chat_completion(
        self, 
        prompt: str, 
        model: str = None, 
        temperature: float = None
    ) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=model or self.config.DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature or self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error in chat completion: {str(e)}")
            return f"Error generating completion: {str(e)}"

    def stream_chat_completion(
        self, 
        prompt: str, 
        model: str = None
    ) -> Generator[str, None, None]:
        try:
            completion = self.client.chat.completions.create(
                model=model or self.config.DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in completion:
                yield chunk.choices[0].delta.content or ""
        except Exception as e:
            self.logger.error(f"Error in streaming completion: {str(e)}")
            yield f"Error streaming completion: {str(e)}"

    def langchain_query(
        self, 
        prompt: str, 
        model: str = None
    ) -> str:
        try:
            original_model = self.langchain_llm.model_name
            if model:
                self.langchain_llm.model_name = model

            response = self.langchain_llm.invoke(prompt)
            
            if model:
                self.langchain_llm.model_name = original_model

            return response.content
        except Exception as e:
            self.logger.error(f"Error with Langchain query: {str(e)}")
            return f"Error with Langchain query: {str(e)}"

    def multi_turn_conversation(
        self, 
        messages: List[Dict[str, str]]
    ) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.config.DEFAULT_MODEL,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error in multi-turn conversation: {str(e)}")
            return f"Conversation error: {str(e)}"

    @classmethod
    def get_available_models(cls) -> List[str]:
        return Config.AVAILABLE_MODELS