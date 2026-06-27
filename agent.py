from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from typing import TypedDict, List, Dict
import pandas as pd
from operator import add #ajouter à chaque fois un nouveau état
from typing_extensions import Annotated, TypedDict
from qdrant_client.models import VectorParams, Distance
from langchain.tools import tool
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import ChatOllama
from dotenv import  load_dotenv
from qdrant_client.models import (
    Filter,
    FieldCondition,
    GeoRadius,
    GeoPoint
    )
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings

#####   1 - Construction de la source de données et Prétraitement
#chargement des variable d'environnement, clé open AI
load_dotenv()

#crteation du modèle
model = ChatOllama(
    model="llama3.2", 
    temperature=0
)
df = pd.read_csv(
    "datatourisme-tour.csv", 
    sep=",", 
    encoding="utf-8"
)
#Les coordonnées sont essentielles.
df = df[df["Latitude"].notna()]
df = df[df["Longitude"].notna()]

#Vérifier aussi les valeurs aberrantes :
#Latitude : 41°N → 51.5°N Longitude : -5° → 10°
df = df[
(df["Latitude"] > 41) &
(df["Latitude"] < 52)
] 

df = df[
(df["Longitude"] > -5) &
(df["Longitude"] < 11)
] # Pour la France métropolitaine.

df = df.drop_duplicates() # supprime les lignes identique
df = df.drop_duplicates(subset=["Nom_du_POI", "Latitude", "Longitude"])

#supprimer les lignes inutilisables, Un POI sans nom n'a aucun intérêt.
df = df.dropna(
    subset=["Nom_du_POI"]
)
# remplacer les valeurs manquantes
df["Description"] = df["Description"].fillna("")

def build_document(row) :
    return f"""
Nom_du_POI : {row['Nom_du_POI']}
Categories_de_POI : {row['Categories_de_POI']}
Adresse_postale : {row['Adresse_postale']}
Code_postal_et_commune: {row['Code_postal_et_commune']}
Description: {row['Description']}
""".strip()
def build_metadata(row):
    metadata = {
        "Nom_du_POI" : row['Nom_du_POI'], 
        "Categories_de_POI" : row['Categories_de_POI'], 
         "location": {
            "lat": row["Latitude"],
            "lon": row["Longitude"]
        },
        "Code_postal_et_commune" : row['Code_postal_et_commune']
       

}
    return metadata
df["document"] = df.apply(
    build_document,
    axis=1
)

df["metadata"] = df.apply(
    build_metadata,
    axis=1
)

documents = df["document"].tolist()
metadatas = df["metadata"].tolist()

#Convertir en Documents
from langchain_core.documents import Document
docs_metadata = [
    Document(
        page_content=doc,
        metadata=meta
    )
    for doc, meta in zip(df["document"], df["metadata"])
]
##### 2- vectorisation. 
#connexion 
client = QdrantClient(
    url="http://localhost:6333"
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


client.recreate_collection(
    collection_name="tourisme_tour_france",
    vectors_config=VectorParams(
        size=384,  # IMPORTANT: dépend de ton modèle d'embedding
        distance=Distance.COSINE
    )
)

#creation de vector_store persistant
v_stores = QdrantVectorStore(
        client=client,
        collection_name="tourisme_tour_france",
        embedding=embeddings
    )
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs_metadata)
print(len(all_splits))
from qdrant_client.models import Distance, VectorParams

collections = client.get_collections().collections
existing = [c.name for c in collections]

if "tourisme_tour_france" not in existing:
    client.recreate_collection    (
        collection_name="tourisme_tour_france",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )
#v_stores.add_documents(all_splits)
##### 3- Développement des outils 

# Tool to search for POI - Tool 1 : recherche vectorielle Qdrant
@tool
def search_poi(query:str ) : 
    """
     Recherche des points d'intérêt touristiques dans Qdrant. 
    """
    docs = v_stores.similarity_search(query, k=5)

    return[
        {
            "Nom_du_POI": d.metadata.get("Nom_du_POI"),
            "Categories_de_POI": d.metadata.get("Categories_de_POI"),
            "Adresse_postale": d.metadata.get("Adresse_postale"), 
            "Code_postal_et_commune": d.metadata.get("Code_postal_et_commune"), 
            "Description": d.metadata.get("Description") #d.page_content
        }
        for d in docs
    ] 

@tool
def search_nearby(latitude: float, longitude: float, radius_km: float = 10, limit: int = 10):
    """Chercher dans le cadre du tourisme en france les point d'intérêt à partir de la géolocalisation """
    points, _ = client.scroll(
        collection_name="tourisme_tour_france",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="location",
                    geo_radius=GeoRadius(
                        center=GeoPoint(
                            lat=latitude,
                            lon=longitude
                        ),
                        radius=radius_km * 1000  # mètres
                    )
                )
            ]
        ),
        limit=limit,
        with_payload=True
    )

    return [
          {
            "Nom_du_POI": p.payload.get("Nom_du_POI"),
            "Categories_de_POI": p.payload.get("Categories_de_POI"),
            "Adresse_postale" : p.payload.get("Adresse_postale"), 
            "Code_postal_et_commune": p.payload.get("Code_postal_et_commune"),
            "Latitude": p.payload.get("Latitude"),
            "Longitude": p.payload.get("Longitude"),
            "Description": p.payload.get("Description")
        }
        for p in points
    ]

@tool
def format_poi (pois) : 
    """formatage des POIs"""
    return "\n".join([
        f"{p['Nom_du_POI']} - {p['Code_postal_et_commune']} ({p['Description']})"
        for p in pois
    ])

tools = [search_poi, search_nearby, format_poi]
#on va utiliser uen fonctions bindtool pour lier le, modèle avec les toolsq qu'il peut utiliser
tools_by_name = {tool.name: tool for tool in tools}
#on passe au modèle une liste des tools qui seront utilisés par ce dernier
model_with_tools = model.bind_tools(tools)
##### 4- Développement de l’architecture du graphe LangGraph – mise en œuvre du RAG Agentic. 
###### User Query -> LangGraph Agent -> Routeur(Tools) -> Context structuré (POIs)->  Réponse finale du LLM
##### 4.1-défnition du State

class State(TypedDict):
    query: str
    messages : Annotated[list, add]
    pois : list[Dict]
    answer: str
    steps: int
##### 4.2-Noeud Agent
from langchain_core.messages import HumanMessage, SystemMessage

def agent(state: State):
    messages = [
        SystemMessage(content=
            "Tu es un assistant touristique spécialisé sur la France. "
            "Utilise les outils si nécessaire pour obtenir des informations."
        ),
        HumanMessage(content=state["query"])
    ] + state.get("messages", [])
    response = model_with_tools.invoke(messages)

    return {
        "messages": state.get("messages", []) + [response],
        "steps": state.get("steps", 0) + 1
    }
##### 4.3-  Noeud de Tools
from langgraph.prebuilt import ToolNode
tool_node = ToolNode(tools)
##### 4.4 -Fonction de routage
def should_continue(state):
    if state.get("steps", 0) > 3:
        return "end"
    last_message = state["messages"][-1]
    
    # si tool call → tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # sinon on force final
    return "end"
##### 4.5- Noeud de fin
def finalize(state: State):
    
      # chercher le dernier message du LLM (pas tool)
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and msg.content:
            return {"answer": msg.content}

    return {"answer": "Aucune réponse générée"}
#### 5. Orchestration LangGraph
from langgraph.graph import StateGraph, START,END

builder = StateGraph(State)

builder.add_node("agent", agent)
builder.add_node("tools", tool_node)
builder.add_node("final", finalize)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": "final"
    }
)
builder.add_edge("tools", "agent")
builder.add_edge("final", END)

builder = builder.compile()
#### 6. Exécution
result = builder.invoke({
    "query": "",
    "messages": [],
    "pois": []
})
#print(result)
print(result["answer"])
#df.info()
#Conserver uniquement ce qui apporte du contexte touristique.
#print(df.isnull().sum()) # nombre de valeur manquante
#df.head()
#print(df['Description'].isnull().sum())
#print(df['Description'].isnull().sum())
