import requests
import warnings
from Main import get_retriever

# Ignore Huggingface symlink warnings for cleaner output
warnings.filterwarnings("ignore")

def add_metformin():
    print("Fetching Wikipedia article for Metformin...")
    url = "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles=Metformin"
    headers = {'User-Agent': 'MedicalRetrievalBot/1.0 (test@example.com)'}
    response = requests.get(url, headers=headers).json()
    pages = response['query']['pages']
    page = list(pages.values())[0]
    text = page['extract']
    
    print(f"Fetched {len(text)} characters. Loading models to inject into database...")
    
    retriever, corpus = get_retriever()
    doc_id = retriever.add_document(text=text, question="Wikipedia article on Metformin", label="Wikipedia")
    
    print(f"Successfully chunked and injected into the database!")

if __name__ == "__main__":
    add_metformin()
