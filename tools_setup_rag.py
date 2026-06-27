# concernant la creation d'uin Agent React avec langgraph
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






# Tool to search for POI - Tool 1 : recherche vectorielle Qdrant
@tool
def search_poi(query:str ) : 
    """
     Recherche des points d'intérêt touristiques dans Qdrant. 
    """
    docs = v_stores.similarity_search(query, k=5)

    return[
        {
            "Nom_du_POI": d.payload.get("Nom_du_POI"),
            "Categories_de_POI": d.payload.get("Categories_de_POI"),
            "Adresse_postale": d.payload.get("Adresse_postale"), 
            "Code_postal_et_commune": d.payload.get("Code_postal_et_commune"), 
            "Description": d.payload.get("Description")
        }
        for d in docs
    ] 

@tool
def search_nearby(latitude: float, longitude: float, radius_km: float = 10, limit: int = 10):
    """Chercher dans le cadre du tourisme en france les point d'intérêt à partir de la géolocalisation """
    results = client.scroll(
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
        for p in results
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