import requests
import json
import uuid
import argparse
from typing import Optional

def test_set_context(
    base_url: str = 'http://64.227.138.80:5000', 
    user_id: Optional[str] = None, 
    context: Optional[str] = None, 
    chunk_size: int = 300
):
    """
    Test the set_context endpoint with flexible parameters
    
    Args:
        base_url (str): Base URL of the API
        user_id (str, optional): User ID for the context
        context (str, optional): Text context to send
        chunk_size (int): Size of text chunks
    """
    # Generate a random user ID if not provided
    if user_id is None:
        user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    
    # Use a default context if not provided
    if context is None:
        context = """
        Cancellation Policy:
 we aim to provide flexibility and convenience for our customers. You can cancel your order anytime before it is shipped by contacting our support team at [support email or phone number]. Once the order is shipped, cancellations will not be possible, but you can still initiate a return after receiving the product, as per our return policy.

Refunds for canceled orders will be processed immediately, and the amount will be credited to your original payment method within [refund processing time, e.g., 7-10 business days]. For assistance with cancellations, feel free to reach out to us.
        Return Policy:
 we strive to ensure customer satisfaction. If you are not completely satisfied with your purchase, you can return the product within [specific return period, e.g., 30 days] from the date of delivery, provided it is in its original condition, unused, and in the original packaging. Please include the receipt or proof of purchase. Refunds will be processed upon inspection of the returned item, and the amount will be credited to your original payment method within [refund processing time, e.g., 7-10 business days]. For more details, please contact our support team at [support email or phone number]. """
    
    # Prepare the payload
    payload = {
        "user_id": "u13",
        "context": context,
        "chunk_size": chunk_size
    }
    
    # Headers for the request
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        # Send POST request
        response = requests.post(
            f"{base_url}/set_context", 
            data=json.dumps(payload), 
            headers=headers
        )
        
        # Print detailed response information
        print("\n--- Response Details ---")
        print(f"Status Code: {response.status_code}")
        print("Response Headers:")
        for key, value in response.headers.items():
            print(f"{key}: {value}")
        
        # Try to parse and print JSON response
        try:
            response_json = response.json()
            print("\nResponse Body:")
            print(json.dumps(response_json, indent=2))
        except json.JSONDecodeError:
            print("\nResponse Text:")
            print(response.text)
        
        # Validate response
        if response.status_code == 200:
            print("\n✅ Context set successfully!")
        else:
            print(f"\n❌ Error setting context. Status code: {response.status_code}")
    
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

def main():
    # Setup argument parsing
    parser = argparse.ArgumentParser(description='Test set_context endpoint')
    parser.add_argument('--url', default='http://64.227.138.80:5000', 
                        help='Base URL of the API')
    parser.add_argument('--user_id', help='Specific user ID to use')
    parser.add_argument('--chunk_size', type=int, default=300, 
                        help='Chunk size for context')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run test with provided or default parameters
    test_set_context(
        base_url=args.url, 
        user_id=args.user_id, 
        chunk_size=args.chunk_size
    )

if __name__ == "__main__":
    main()