# CallWise API Service

This repository contains the backend service for the **CallWise** platform, built using Flask. The service includes features such as user authentication with MongoDB, integration with SambaNova APIs, and endpoints for setting API keys, models, context, and performing RAG (Retrieve and Generate) queries.

## Features

* MongoDB connection for user and context data
* API key management for integrating SambaNova services
* RAG query handling for contextual responses
* Context embedding with chunking for efficient information retrieval
* Model selection for specific user tasks
* Comprehensive logging for easier debugging
* Health check endpoint to monitor the service's status

## Prerequisites

* **Python** (>= 3.8)
* **MongoDB** (connected via URI in `.env`)
* **Flask** and required dependencies (see `requirements.txt`)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/callwise-backend.git
cd callwise-backend
```

### 2. Install Dependencies

Create a virtual environment and install the required Python packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set Environment Variables

Create a `.env` file in the project directory and define the following variable:

```env
URI=<your-mongodb-connection-uri>
```

### 4. Run the Server

Start the Flask development server:

```bash
python app.py
```

## Endpoints

### Health Check

**GET** `/health`
Checks the health of the server.

Response:
```json
{
    "status": "healthy",
    "version": "1.0.0"
}
```

### Test MongoDB Connection

**POST** `/test_mongo`
Tests if MongoDB is connected by inserting a sample document.

Response:
```json
{
    "status": "success",
    "message": "MongoDB connected and test document added!"
}
```

### Set API Key

**POST/GET** `/set_api_key`
Saves the user's API key for SambaNova and retrieves available models.

Request Body:
```json
{
    "user_id": "user123",
    "api_key": "sambanova-api-key"
}
```

Response:
```json
{
    "message": "API key set successfully",
    "available_models": ["model1", "model2"],
    "status": "ok"
}
```

### Get Available Models

**GET** `/get_models`
Fetches available models for the given user.

Query Parameter:
- `user_id`

Response:
```json
{
    "models": ["model1", "model2"],
    "status": "ok"
}
```

### Set Model

**POST** `/set_model`
Assigns a specific model to a user.

Request Body:
```json
{
    "user_id": "user123",
    "model": "model1"
}
```

Response:
```json
{
    "message": "Model set successfully",
    "model": "model1"
}
```

### Set Context

**POST** `/set_context`
Saves user context with chunking for embedding.

Request Body:
```json
{
    "user_id": "user123",
    "context": "Your context data goes here.",
    "chunk_size": 300
}
```

Response:
```json
{
    "message": "Context set successfully",
    "context": "Your context data goes here."
}
```

### RAG Query

**POST** `/rag_query`
Processes a Retrieve-and-Generate (RAG) query based on user context.

Request Body:
```json
{
    "user_id": "user123",
    "query": "Your query here."
}
```

Response:
```json
{
    "response": "Generated response from SambaNova model."
}
```

## Logging

Logs are configured to output important messages and errors for debugging:
* Level: `INFO`
* Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

## Deployment

To deploy this service:

1. Configure your deployment platform (e.g., AWS, Heroku, Docker) to use the provided `app.py`
2. Ensure the environment variables (e.g., `URI`) are set correctly

## License

This project is licensed under the MIT License.
