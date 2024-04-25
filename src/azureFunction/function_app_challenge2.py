import re
import azure.functions as func
import datetime
import json
import logging
import os
from openai import AzureOpenAI

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.vector_stores.azureaisearch import IndexManagement
from llama_index.core.settings import Settings
from llama_index.readers.azstorage_blob import AzStorageBlobReader
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

app = func.FunctionApp()

@app.function_name(name="AskQuestion")
@app.route(route="AskQuestion", auth_level=func.AuthLevel.ANONYMOUS)
@app.cosmos_db_input(arg_name="inputDocuments", 
                     database_name="aoaidb",
                     container_name="facts",
                     connection="MyAccount_COSMOSDB")
def AskQuestion(inputDocuments: func.DocumentList, req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Get the question from the request parameters
    raw_question = req.params.get('question')

    # Check if the question is provided
    if not raw_question:
        return func.HttpResponse(
            "Please pass a question on the query string",
            status_code=400
        )

    # Remove all HTML tags from the question
    clean_question = re.sub('<.*?>', '', raw_question)

    if inputDocuments:
        # TODO - Create an AI search client to connect to the resource
        searchClient = SearchClient(
            endpoint=os.getenv("AISearchEndpoint"),
            index_name=os.getenv("AISearchIndexName"),
            credential=AzureKeyCredential(os.getenv("AISearchAPIKey"))
        )
        
        client = AzureOpenAI(
            azure_endpoint = os.getenv("AOAI_ENDPOINT"), 
            api_key=os.getenv("AOAI_KEY"),  
            api_version="2024-02-15-preview"
        )

        #Join all the facts into a single string
        facts = "These are the local experts." + "\n".join([doc.data['fact'] for doc in inputDocuments])                        
           
        # TODO - Create a text embedding of the question from the OpenAI Model.
        embedding = client.embeddings.create(input=clean_question, model="text-embedding-ada-amd").data[0].embedding
        logging.info(f'Embedding: {embedding}')

        # TODO - Create a vectorized query to sent to AI Search
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=3, fields="embedding")

        # TODO - Join all the facts from the AI search results into a single string
        # TODO - Add the AI search results to the facts
        ai_search_results = searchClient.search(  
            search_text=clean_question,  
            vector_queries= [vector_query],
            select=["chunk"],
            top=5
        )

        ai_facts = "\n".join([result["chunk"] for result in ai_search_results])

        facts = facts + "\n" + ai_facts

        message_text = [{"role":"system","content": facts}, 
                        {"role":"user","content": clean_question + 
                         ". Be as helpful as possible in connecting the above local experts in the response. State the response as a gramatically correct and complete summary."}]

        completion = client.chat.completions.create(
            messages = message_text,
            model = os.environ.get("MODEL", "gpt35"),
            temperature = float(os.environ.get("TEMPERATURE", "0.7")), 
            max_tokens = int(os.environ.get("MAX_TOKENS", "800")),
            top_p = float(os.environ.get("TOP_P", "0.95")),
            frequency_penalty = float(os.environ.get("FREQUENCY_PENALTY", "0")),
            presence_penalty = float(os.environ.get("PRESENCE_PENALTY", "0")),
            stop = os.environ.get("STOP", "None")
        )

    return func.HttpResponse(
        completion.choices[0].message.content,
        status_code=200
        )
