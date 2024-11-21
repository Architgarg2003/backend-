# import os
# import logging
# from flask import Flask, request, jsonify, Response
# from werkzeug.exceptions import BadRequest
# from sambanova_client import SambaNovaClient

# # Flask App Configuration
# app = Flask(__name__)

# # Logging Configuration
# logging.basicConfig(
#     level=logging.INFO, 
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Global configuration class to manage application state
# class AppConfig:
#     def __init__(self):
#         self.sambanova_client = None
#         self.api_key = None
#         self.selected_model = None
#         self.available_models = []

# # Initialize global app configuration
# app_config = AppConfig()

# # Logging Middleware
# @app.before_request
# def log_request_info():
#     """Log request details before processing"""
#     try:
#         logger.info(f'Request Method: {request.method}')
#         logger.info(f'Request Path: {request.path}')
#         logger.info(f'Request Headers: {dict(request.headers)}')
        
#         # Safely get JSON data without raising an error
#         request_data = request.get_json(silent=True)
#         if request_data:
#             logger.info(f'Request Body: {request_data}')
#     except Exception as e:
#         logger.error(f"Error logging request info: {str(e)}")

# @app.after_request
# def log_response_info(response):
#     """Log response details after processing"""
#     try:
#         logger.info(f'Response Status: {response.status}')
#     except Exception as e:
#         logger.error(f"Error logging response info: {str(e)}")
#     return response

# # Error Handlers
# @app.errorhandler(BadRequest)
# def handle_bad_request(e):
#     """Handle bad request errors"""
#     logger.error(f"Bad Request: {str(e)}")
#     return jsonify({"error": "Bad request", "details": str(e)}), 400

# @app.errorhandler(500)
# def handle_server_error(e):
#     """Handle internal server errors"""
#     logger.error(f"Server Error: {str(e)}")
#     return jsonify({"error": "Internal server error", "details": str(e)}), 500

# # API Routes
# @app.route('/set_api_key', methods=['POST'])
# def set_api_key():
#     """Set SambaNova API key"""
#     # Check if API key is already set
#     if app_config.api_key is not None:
#         return jsonify({
#             "error": "API key already set",
#             "status": "Initialized"
#         }), 400
    
#     # Parse request data
#     data = request.json
    
#     if not data:
#         return jsonify({"error": "Request body is required"}), 400
    
#     api_key = data.get('api_key')
    
#     if not api_key:
#         return jsonify({"error": "API key is required"}), 400
    
#     try:
#         # Validate API key by creating a test client
#         test_client = SambaNovaClient(api_key)
        
#         # Store API key
#         app_config.api_key = api_key
        
#         # Fetch and store available models
#         app_config.available_models = SambaNovaClient.get_available_models()
        
#         return jsonify({
#             "message": "API key set successfully",
#             "available_models": app_config.available_models
#         }), 200
    
#     except Exception as e:
#         logger.error(f"Error setting API key: {str(e)}")
#         return jsonify({
#             "error": "Failed to set API key", 
#             "details": str(e)
#         }), 500

# @app.route('/set_model', methods=['POST'])
# def set_model():
#     """Set the SambaNova model"""
#     # Check if API key is set first
#     if app_config.api_key is None:
#         return jsonify({
#             "error": "API key must be set first",
#             "status": "Waiting for API key"
#         }), 400
    
#     # Check if client is already initialized
#     if app_config.sambanova_client is not None:
#         return jsonify({
#             "error": "Client already initialized",
#             "current_model": app_config.selected_model
#         }), 400
    
#     # Parse request data
#     data = request.json
    
#     if not data:
#         return jsonify({"error": "Request body is required"}), 400
    
#     # Extract model
#     model = data.get('model')
    
#     # Validate model
#     if not model or model not in app_config.available_models:
#         return jsonify({
#             "error": "Invalid or missing model",
#             "available_models": app_config.available_models
#         }), 400
    
#     try:
#         # Create SambaNova client with stored API key
#         client = SambaNovaClient(app_config.api_key)
        
#         # Store configuration
#         app_config.sambanova_client = client
#         app_config.selected_model = model
        
#         return jsonify({
#             "message": "Model set successfully",
#             "model": model,
#             "status": "Fully Initialized"
#         }), 200
    
#     except Exception as e:
#         logger.error(f"Model initialization error: {str(e)}")
#         return jsonify({
#             "error": "Failed to initialize model", 
#             "details": str(e)
#         }), 500

# @app.route('/models', methods=['GET'])
# def list_available_models():
#     """List available SambaNova models"""
#     try:
#         # Use stored models if available, otherwise fetch
#         models = (app_config.available_models 
#                   if app_config.available_models 
#                   else SambaNovaClient.get_available_models())
        
#         return jsonify({
#             "available_models": models,
#             "total_models": len(models)
#         }), 200
#     except Exception as e:
#         logger.error(f"Error listing models: {str(e)}")
#         return jsonify({
#             "error": "Failed to retrieve models", 
#             "details": str(e)
#         }), 500

# @app.route('/chat', methods=['POST'])
# def chat_completion():
#     """Generate chat completion"""
#     # Check if client is initialized
#     if app_config.sambanova_client is None:
#         return jsonify({"error": "SambaNova client not initialized"}), 401
    
#     # Parse request data
#     data = request.json
    
#     # Validate input
#     if not data or 'prompt' not in data:
#         return jsonify({"error": "Prompt is required"}), 400
    
#     # Extract parameters
#     prompt = data['prompt']
#     query_type = data.get('query_type', 'direct')
    
#     try:
#         # Generate response based on query type
#         if query_type == 'direct':
#             response = app_config.sambanova_client.generate_chat_completion(
#                 prompt=prompt, 
#                 model=app_config.selected_model
#             )
#         elif query_type == 'langchain':
#             response = app_config.sambanova_client.langchain_query(
#                 prompt=prompt, 
#                 model=app_config.selected_model
#             )
#         else:
#             return jsonify({"error": "Invalid query type"}), 400
        
#         return jsonify({
#             "response": response,
#             "model": app_config.selected_model
#         }), 200
    
#     except Exception as e:
#         logger.error(f"Chat completion error: {str(e)}")
#         return jsonify({
#             "error": "Failed to generate response", 
#             "details": str(e)
#         }), 500

# @app.route('/stream', methods=['POST'])
# def stream_chat_completion():
#     """Stream chat completion response"""
#     # Check if client is initialized
#     if app_config.sambanova_client is None:
#         return jsonify({"error": "SambaNova client not initialized"}), 401
    
#     # Parse request data
#     data = request.json
    
#     # Validate input
#     if not data or 'prompt' not in data:
#         return jsonify({"error": "Prompt is required"}), 400
    
#     # Extract parameters
#     prompt = data['prompt']
    
#     def generate():
#         try:
#             for chunk in app_config.sambanova_client.stream_chat_completion(prompt, app_config.selected_model):
#                 yield f"data: {chunk}\n\n"
#         except Exception as e:
#             yield f"data: {str(e)}\n\n"
    
#     return Response(generate(), mimetype='text/event-stream')

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         "status": "healthy",
#         "version": "1.0.0",
#         "api_key_set": app_config.api_key is not None,
#         "model_set": app_config.selected_model is not None,
#         "client_initialized": app_config.sambanova_client is not None
#     }), 200

# @app.route('/set_context', methods=['POST'])
# def set_context():
#     """
#     Set context with embedding
#     """
#     data = request.json
    
#     if not data or 'context' not in data:
#         return jsonify({"error": "Context is required"}), 400
    
#     try:
#         # Set context and create embeddings
#         app_config.sambanova_client.set_rag_context(
#             data['context'], 
#             data.get('chunk_size', 300)
#         )
        
#         # Get context statistics
#         context_stats = app_config.sambanova_client.rag_assistant.get_context_stats()
        
#         return jsonify({
#             "message": "Context set with embeddings",
#             "stats": context_stats
#         }), 200
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/rag_query', methods=['POST'])
# def rag_query():
#     """
#     Handle RAG query with embedded context
#     """
#     data = request.json
    
#     if not data or 'query' not in data:
#         return jsonify({"error": "Query is required"}), 400
    
#     try:
#         # Get RAG response
#         response = app_config.sambanova_client.get_rag_response(data['query'])
        
#         return jsonify({
#             "response": response
#         }), 200
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5001)

'''working code for one user at a time'''







# import os
# import logging
# from flask import Flask, request, jsonify, Response
# from werkzeug.exceptions import BadRequest
# from sambanova_client import SambaNovaClient
# from flask_pymongo import PyMongo
# from flask_cors import CORS

# # Flask App Configuration
# app = Flask(__name__)
# # Set MongoDB URI
# from pymongo.mongo_client import MongoClient
# from pymongo.server_api import ServerApi

# # Replace <db_password> with your actual MongoDB password
# uri = "mongodb+srv://architgarg2003:Iambornin2003@cluster0.usvrd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# # Create a new client and connect to the server
# mongo = MongoClient(uri, server_api=ServerApi('1'))

# # Send a ping to confirm a successful connection
# try:
#     mongo.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print("Failed to connect to MongoDB:")
#     print(e)
# CORS(app)  # Enable CORS for all routes

# # Logging Configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# @app.route('/test_mongo', methods=['POST'])
# def test_mongo():
#     try:
#         # Access a collection (use 'test' as default for testing)
#         mongo.db.insert_one({"message": "Hello, MongoDB!"})
#         return {"status": "success", "message": "MongoDB connected and test document added!"}, 200
#     except Exception as e:
#         return {"status": "error", "message": str(e)}, 500
    

# # API Routes
# @app.route('/set_api_key', methods=['POST'])
# def set_api_key():
#     """Set SambaNova API key for a user"""
#     user_id = request.json.get('user_id')
#     api_key = request.json.get('api_key')

#     if not user_id or not api_key:
#         return jsonify({"error": "User  ID and API key are required"}), 400

#     # Check if the user already exists
#     user_data = mongo.db.users.find_one({"user_id": user_id})

#     if user_data:
#         # If the user exists, update the API key
#         mongo.db.users.update_one({"user_id": user_id}, {"$set": {"api_key": api_key}})
#     else:
#         # If the user does not exist, create a new user with the API key
#         mongo.db.users.insert_one({"user_id": user_id, "api_key": api_key, "available_models": []})

#     # Fetch and store available models
#     available_models = SambaNovaClient.get_available_models()
#     mongo.db.users.update_one({"user_id": user_id}, {"$set": {"available_models": available_models}})

#     return jsonify({
#         "message": "API key set successfully",
#         "available_models": available_models
#     }), 200


# @app.route('/set_model', methods=['POST'])
# def set_model():
#     """Set the SambaNova model for a user"""
#     user_id = request.json.get('user_id')
#     model = request.json.get('model')

#     if not user_id or not model:
#         return jsonify({"error": "User  ID and model are required"}), 400

#     user_data = mongo.db.users.find_one({"user_id": user_id})

#     if not user_data or 'api_key' not in user_data:
#         return jsonify({"error": "API key must be set first"}), 400

#     if 'sambanova_client' in user_data:
#         return jsonify({"error": "Client already initialized", "current_model": user_data.get('selected_model')}), 400

#     try:
#         client = SambaNovaClient(user_data['api_key'])
#         mongo.db.users.update_one({"user_id": user_id}, {"$set": {"sambanova_client": client, "selected_model": model}})
#         return jsonify({
#             "message": "Model set successfully",
#             "model": model,
#             "status": "Fully Initialized"
#         }), 200
#     except Exception as e:
#         logger.error(f"Model initialization error: {str(e)}")
#         return jsonify({"error": "Failed to initialize model", "details": str(e)}), 500

# @app.route('/set_context', methods=['POST'])
# def set_context():
#     """Set context with embedding for a user"""
#     user_id = request.json.get('user_id')
#     context = request.json.get('context')
#     chunk_size = request.json.get('chunk_size', 300)

#     if not user_id or not context:
#         return jsonify({"error": "User  ID and context are required"}), 400

#     user_data = mongo.db.users.find_one({"user_id": user_id})

#     if not user_data or 'sambanova_client' not in user_data:
#         return jsonify({"error": "Client not initialized"}), 401

#     try:
#         user_data['sambanova_client'].set_rag_context(context, chunk_size)
#         context_stats = user_data['sambanova_client'].rag_assistant.get_context_stats()

#         return jsonify({
#             "message": "Context set with embeddings",
#             "stats": context_stats
#         }), 200
#     except Exception as e:
#         logger.error(f"Error setting context: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/rag_query', methods=['POST'])
# def rag_query():
#     """Handle RAG query with embedded context for a user"""
#     user_id = request.json.get('user_id')
#     query = request.json.get('query')

#     if not user_id or not query:
#         return jsonify({"error": "User  ID and query are required"}), 400

#     user_data = mongo.db.users.find_one({"user_id": user_id})

#     if not user_data or 'sambanova_client' not in user_data:
#         return jsonify({"error": "Client not initialized"}), 401

#     try:
#         response = user_data['sambanova_client'].get_rag_response(query)
#         return jsonify({
#             "response": response
#         }), 200
#     except Exception as e:
#         logger.error(f"Error handling RAG query: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         "status": "healthy",
#         "version": "1.0.0"
#     }), 200

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5001)





# import os
# import logging
# from flask import Flask, request, jsonify
# from flask_pymongo import PyMongo
# from flask_cors import CORS
# from sambanova_client import SambaNovaClient
# from pymongo.mongo_client import MongoClient
# from pymongo.server_api import ServerApi

# # Flask App Configuration
# app = Flask(__name__)

# # Set MongoDB URI
# uri = "mongodb+srv://architgarg2003:Iambornin2003@cluster0.usvrd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# mongo = MongoClient(uri, server_api=ServerApi('1'))

# # Send a ping to confirm a successful connection
# try:
#     mongo.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print("Failed to connect to MongoDB:")
#     print(e)

# CORS(app)  # Enable CORS for all routes

# # Logging Configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# @app.route('/test_mongo', methods=['POST'])
# def test_mongo():
#     try:
#         # Access a collection (use 'test' as default for testing)
#         mongo.db.test.insert_one({"message": "Hello, MongoDB!"})
#         return {"status": "success", "message": "MongoDB connected and test document added!"}, 200
#     except Exception as e:
#         return {"status": "error", "message": str(e)}, 500
    

# # API Routes
# @app.route('/set_api_key', methods=['POST'])
# def set_api_key():
#     """Set SambaNova API key for a user"""
#     user_id = request.json.get('user_id')
#     api_key = request.json.get('api_key')

#     if not user_id or not api_key:
#         return jsonify({"error": "User  ID and API key are required"}), 400

#     # Check if the user already exists
#     user_data = mongo.db.users.find_one({"user_id": user_id})

#     if user_data:
#         # If the user exists, update the API key
#         mongo.db.users.update_one({"user_id": user_id}, {"$set": {"api_key": api_key}})
#     else:
#         # If the user does not exist, create a new user with the API key
#         mongo.db.users.insert_one({"user_id": user_id, "api_key": api_key, "available_models": []})

#     # Fetch and store available models
#     available_models = SambaNovaClient.get_available_models()
#     mongo.db.users.update_one({"user_id": user_id}, {"$set": {"available_models": available_models}})

#     return jsonify({
#         "message": "API key set successfully",
#         "available_models": available_models
#     }), 200


# @app.route('/set_model', methods=['POST'])
# def set_model():
#     """Set the SambaNova model for a user"""
#     user_id = request.json.get('user_id')
#     model = request.json.get('model')

#     if not user_id or not model:
#         return jsonify({"error": "User  ID and model are required"}), 400

#     user_data = mongo.db.users.find_one({"user_id": user_id})

#     if not user_data or 'api_key' not in user_data:
#         return jsonify({"error": "API key must be set first"}), 400

#     # Update the selected model for the user
#     mongo.db.users.update_one({"user_id": user_id}, {"$set": {"selected_model": model}})

#     return jsonify({
#         "message": "Model set successfully",
#         "model": model,
#     }), 200

# @app.route('/set_context', methods=['POST'])
# def set_context():
#     """Set context with embedding for a user"""
#     user_id = request.json.get('user_id')
#     context = request.json.get('context')
#     chunk_size = request.json.get('chunk_size', 300)

#     if not user_id or not context:
#         return jsonify({"error": "User  ID and context are required"}), 400

#     user_data = mongo.db.users.find_one({"user_id": user_id})

#     if not user_data or 'api_key' not in user_data:
#         return jsonify({"error": "API key must be set first"}), 400

#     try:
#         # Initialize SambaNovaClient with user's API key
#         client = SambaNovaClient(user_data['api_key'])
#         client.set_rag_context(context, chunk_size)

#         # Store the context in the database
#         mongo.db.users.update_one({"user_id": user_id}, {"$set": {"context": context}})

#         return jsonify({
#             "message": "Context set successfully",
#             "context": context
#         }), 200
#     except Exception as e:
#         logger.error(f"Error setting context: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# # @app.route('/rag_query', methods=['POST'])
# # def rag_query():
# #     """Handle RAG query with embedded context for a user"""
# #     user_id = request.json.get('user_id')
# #     query = request.json.get('query')

# #     if not user_id or not query:
# #         return jsonify({"error": "User  ID and query are required"}), 400

# #     user_data = mongo.db.users.find_one({"user_id": user_id})

# #     if not user_data or 'api_key' not in user_data:
# #         return jsonify({"error": "API key must be set first"}), 400

# #     try:
# #         # Initialize SambaNovaClient with user's API key
# #         client = SambaNovaClient(user_data['api_key'])
# #         response = client.get_rag_response(query)

# #         return jsonify({
# #             "response": response
# #         }), 200
# #     except Exception as e:
# #         logger.error(f"Error handling RAG query: {str(e)}")
# #         return jsonify({"error": str(e)}), 500


# @app.route('/rag_query', methods=['POST'])
# def rag_query():
#     """Handle RAG query with embedded context for a user"""
#     user_id = request.json.get('user_id')
#     query = request.json.get('query')

#     if not user_id or not query:
#         return jsonify({"error": "User  ID and query are required"}), 400

#     user_data = mongo.db.users.find_one({"user_id": user_id})

#     if not user_data or 'api_key' not in user_data:
#         return jsonify({"error": "API key must be set first"}), 400

#     # Retrieve context from user data
#     context = user_data.get('context')
#     if not context:
#         return jsonify({"error": "No context provided. Please set context first."}), 400

#     try:
#         # # Initialize SambaNovaClient with user's API key
#         client = SambaNovaClient(user_data['api_key'])
#         client.set_rag_context(context)

#         # Get the RAG response
#         response = client.get_rag_response(query)

#         return jsonify({
#             "response": response
#         }), 200
#     except Exception as e:
#         logger.error(f"Error processing RAG query: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         "status": "healthy",
#         "version": "1.0.0"
#     }), 200

# if __name__ == '__main__':
#     # app.run(debug=True, host='0.0.0.0', port=5001)
#     app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))





import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from sambanova_client import SambaNovaClient  # Assuming this class exists similarly to the Flask version

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Set MongoDB URI
uri = "mongodb+srv://architgarg2003:Iambornin2003@cluster0.usvrd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongo = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    mongo.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print("Failed to connect to MongoDB:")
    print(e)

# Pydantic Models for Request Validation
class APIKeyRequest(BaseModel):
    user_id: str
    api_key: str

class ModelRequest(BaseModel):
    user_id: str
    model: str

class ContextRequest(BaseModel):
    user_id: str
    context: str
    chunk_size: int = 300

class RAGQueryRequest(BaseModel):
    user_id: str
    query: str

@app.post("/test_mongo")
async def test_mongo():
    try:
        # Access a collection (use 'test' as default for testing)
        mongo.db.test.insert_one({"message": "Hello, MongoDB!"})
        return {
            "status": "success", 
            "message": "MongoDB connected and test document added!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_api_key")
async def set_api_key(request: APIKeyRequest):
    """Set SambaNova API key for a user"""
    try:
        # Check if the user already exists
        user_data = mongo.db.users.find_one({"user_id": request.user_id})

        if user_data:
            # If the user exists, update the API key
            mongo.db.users.update_one(
                {"user_id": request.user_id}, 
                {"$set": {"api_key": request.api_key}}
            )
        else:
            # If the user does not exist, create a new user with the API key
            mongo.db.users.insert_one({
                "user_id": request.user_id, 
                "api_key": request.api_key, 
                "available_models": []
            })

        # Fetch and store available models
        available_models = SambaNovaClient.get_available_models()
        mongo.db.users.update_one(
            {"user_id": request.user_id}, 
            {"$set": {"available_models": available_models}}
        )

        return {
            "message": "API key set successfully",
            "available_models": available_models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_model")
async def set_model(request: ModelRequest):
    """Set the SambaNova model for a user"""
    try:
        user_data = mongo.db.users.find_one({"user_id": request.user_id})

        if not user_data or 'api_key' not in user_data:
            raise HTTPException(status_code=400, detail="API key must be set first")

        # Update the selected model for the user
        mongo.db.users.update_one(
            {"user_id": request.user_id}, 
            {"$set": {"selected_model": request.model}}
        )

        return {
            "message": "Model set successfully",
            "model": request.model,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_context")
async def set_context(request: ContextRequest):
    """Set context with embedding for a user"""
    try:
        user_data = mongo.db.users.find_one({"user_id": request.user_id})

        if not user_data or 'api_key' not in user_data:
            raise HTTPException(status_code=400, detail="API key must be set first")

        # Initialize SambaNovaClient with user's API key
        client = SambaNovaClient(user_data['api_key'])
        client.set_rag_context(request.context, request.chunk_size)

        # Store the context in the database
        mongo.db.users.update_one(
            {"user_id": request.user_id}, 
            {"$set": {"context": request.context}}
        )

        return {
            "message": "Context set successfully",
            "context": request.context
        }
    except Exception as e:
        logger.error(f"Error setting context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag_query")
async def rag_query(request: RAGQueryRequest):
    """Handle RAG query with embedded context for a user"""
    try:
        user_data = mongo.db.users.find_one({"user_id": request.user_id})

        if not user_data or 'api_key' not in user_data:
            raise HTTPException(status_code=400, detail="API key must be set first")

        # Retrieve context from user data
        context = user_data.get('context')
        if not context:
            raise HTTPException(status_code=400, detail="No context provided. Please set context first.")

        # Initialize SambaNovaClient with user's API key
        client = SambaNovaClient(user_data['api_key'])
        client.set_rag_context(context)

        # Get the RAG response
        response = client.get_rag_response(request.query)

        return {
            "response": response
        }
    except Exception as e:
        logger.error(f"Error processing RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

# Main block to run the application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "app:app", 
        host='127.0.0.1', 
        port=int(os.getenv('PORT', 5001)),
        reload=True
    )