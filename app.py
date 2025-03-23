import os
import logging
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_cors import CORS
from sambanova_client import SambaNovaClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Flask App Configuration
app = Flask(__name__)
CORS(app)

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

  # Enable CORS for all routes

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.route('/test_mongo', methods=['POST'])
def test_mongo():
    try:
        # Access a collection (use 'test' as default for testing)
        mongo.db.test.insert_one({"message": "Hello, MongoDB!"})
        return {"status": "success", "message": "MongoDB connected and test document added!"}, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500
    
@app.route('/set_api_key', methods=['POST','GET'])
def set_api_key():
    """Set SambaNova API key for a user"""
    user_id = request.json.get('user_id')
    api_key = request.json.get('api_key')

    if not user_id or not api_key:
        return jsonify({"error": "User ID and API key are required"}), 400

    try:
        # Check if the user already exists
        user_data = mongo.db.users.find_one({"user_id": user_id})

        if user_data:
            # If the user exists, update the API key
            mongo.db.users.update_one({"user_id": user_id}, {"$set": {"api_key": api_key}})
        else:
            # If the user does not exist, create a new user with the API key
            mongo.db.users.insert_one({
                "user_id": user_id, 
                "api_key": api_key, 
                "available_models": []
            })

        # Fetch and store available models
        available_models = SambaNovaClient.get_available_models()
        
        # Update user with available models
        mongo.db.users.update_one(
            {"user_id": user_id}, 
            {"$set": {"available_models": available_models}}
        )

        # Return successful response with 200 status code
        return jsonify({
            "message": "API key set successfully",
            "available_models": available_models,
            "status": "ok"
        }), 200

    except Exception as e:
        # Log the error for debugging
        print(f"Error setting API key: {str(e)}")
        
        # Return error response
        return jsonify({
            "error": "Failed to set API key",
            "details": str(e),
            "status": "error"
        }), 500
@app.route('/get_models', methods=['GET'])
def get_models():
    """Retrieve available models for a specific user"""
    user_id = request.args.get('user_id')

    if not user_id:
        return jsonify({
            "error": "User ID is required",
            "status": "error"
        }), 400

    try:
        # Find the user in the database
        user_data = mongo.db.users.find_one({"user_id": user_id})

        if not user_data:
            return jsonify({
                "error": "User not found",
                "status": "error"
            }), 404

        # Check if available_models exists and is not empty
        available_models = user_data.get('available_models', [])

        if not available_models:
            # If no models are stored, attempt to fetch models again
            try:
                available_models = SambaNovaClient.get_available_models()
                
                # Update the user's available models in the database
                mongo.db.users.update_one(
                    {"user_id": user_id}, 
                    {"$set": {"available_models": available_models}}
                )
            except Exception as fetch_error:
                return jsonify({
                    "error": "Failed to fetch available models",
                    "details": str(fetch_error),
                    "status": "error"
                }), 500

        return jsonify({
            "models": available_models,
            "status": "ok"
        }), 200

    except Exception as e:
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e),
            "status": "error"
        }), 500
    


@app.route('/set_model', methods=['POST'])
def set_model():
    """Set the SambaNova model for a user"""
    user_id = request.json.get('user_id')
    model = request.json.get('model')

    if not user_id or not model:
        return jsonify({"error": "User  ID and model are required"}), 400

    user_data = mongo.db.users.find_one({"user_id": user_id})

    if not user_data or 'api_key' not in user_data:
        return jsonify({"error": "API key must be set first"}), 400

    # Update the selected model for the user
    mongo.db.users.update_one({"user_id": user_id}, {"$set": {"selected_model": model}})

    return jsonify({
        "message": "Model set successfully",
        "model": model,
    }), 200

@app.route('/set_context', methods=['POST'])
def set_context():
    """Set context with embedding for a user"""
    user_id = request.json.get('user_id')
    context = request.json.get('context')
    chunk_size = request.json.get('chunk_size', 300)

    if not user_id or not context:
        return jsonify({"error": "User  ID and context are required"}), 400

    user_data = mongo.db.users.find_one({"user_id": user_id})

    if not user_data or 'api_key' not in user_data:
        return jsonify({"error": "API key must be set first"}), 400

    try:
        # Initialize SambaNovaClient with user's API key
        client = SambaNovaClient(user_data['api_key'])
        client.set_rag_context(context, chunk_size)

        # Store the context in the database
        mongo.db.users.update_one({"user_id": user_id}, {"$set": {"context": context}})

        return jsonify({
            "message": "Context set successfully",
            "context": context
        }), 200
    except Exception as e:
        logger.error(f"Error setting context: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/rag_query', methods=['POST'])
def rag_query():
    """Handle RAG query with embedded context for a user"""
    user_id = request.json.get('user_id')
    query = request.json.get('query')

    if not user_id or not query:
        return jsonify({"error": "User  ID and query are required"}), 400

    user_data = mongo.db.users.find_one({"user_id": user_id})

    if not user_data or 'api_key' not in user_data:
        return jsonify({"error": "API key must be set first"}), 400

    # Retrieve context from user data
    context = user_data.get('context')
    if not context:
        return jsonify({"error": "No context provided. Please set context first."}), 400

    try:
        # # Initialize SambaNovaClient with user's API key
        client = SambaNovaClient(user_data['api_key'])
        client.set_rag_context(context)

        # Get the RAG response
        response = client.get_rag_response(query)

        return jsonify({
            "response": response
        }), 200
    except Exception as e:
        logger.error(f"Error processing RAG query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0"
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    # app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5001)))




