# pip install sparqlwrapper
# https://rdflib.github.io/sparqlwrapper/

from SPARQLWrapper import SPARQLWrapper, JSON
import re
import urllib.request

endpoint_url = "https://query.wikidata.org/sparql"

# occupation: Q639669 musician
# occupation: Q131524 entrepreneur
# occupation: Q33999 actor
# occupation: Q82955 politician
# occupation: Q82594 computer_scientist
# occupation: Q39631 physician
# occupation: Q483501 artist 7


# gender: Q6581097 Male
# gender: Q6581072 Female

def get_results(endpoint_url, query):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

#occupations = {"Q4964182": "philosopher", "Q36180": "writer"}

#occupations = {"Q43845": "businessperson", "Q639669": "musician", "Q131524": "entrepreneur", "Q33999": "actor"}

#occupations = {"Q82955": "politician", "Q82594": "computer_scientist", "Q39631": "physician", "Q483501": "artist"}

#occupations = {"Q3665646": "basketball", "Q11303721": "golf", "Q10833314": "tennis", "Q937857": "soccer", "Q11607585": "fighter"}

# occupations = {"Q43845": "businessperson", "Q189290":"military_officer", \
#                "Q189290": "judge", "Q3282637": "film_producer", \
#                "Q158852": "conductor", "Q1028181": "painter", \
#                "Q82594": "computer_scientist", "Q39631": "physician", \
#                "Q3665646": "basketball", "Q11303721": "golf", \
#                "Q10833314": "tennis", "Q937857": "soccer", \
#                "Q11607585": "fighter", "Q36180": "writer"
#                 }

#occupations = {"Q10833314": "tennis_player", "Q2462658": "manager"}

#occupations = {"Q41583": "coach"}

#occupations = {"Q40348": "lawyer", "Q639669": "musician"}

#occupations = {"Q158852": "conductor", "Q33999": "actor"}

occupations = {"Q33999": "entrepreneur"}

# for k, v in occupations.items():
#     occupations[k] = v + "_f"


for key, value in occupations.items():

    # query = """SELECT distinct ?item ?label ?img
    # WHERE{
    #
    #       ?item wdt:P106 wd:%s.
    #
    #       ?item rdfs:label ?label.
    #       ?item wdt:P18 ?img.
    #       ?item wdt:P21 wd:Q6581072.
    #
    #       ?item p:P569/psv:P569 ?birth_date_node.
    #       ?birth_date_node wikibase:timePrecision "9"^^xsd:integer .
    #       ?birth_date_node wikibase:timeValue ?birth_date .
    #       FILTER (year(?birth_date) > 1920).
    #
    #       FILTER (LANG(?label) = 'en') .
    #       SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }.
    #     } LIMIT 500"""

    # query = """SELECT distinct ?item ?label ?img
    # WHERE{
    #
    #       ?item wdt:P106 wd:%s.
    #
    #       ?item rdfs:label ?label.
    #       ?item wdt:P18 ?img.
    #       ?item wdt:P21 wd:Q6581072.
    #
    #
    #
    #       FILTER (LANG(?label) = 'en') .
    #       SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }.
    #     } LIMIT 500"""

    query = """SELECT distinct ?item ?label ?img
    WHERE{

          ?item wdt:P106 wd:%s.

          ?item rdfs:label ?label.
          ?item wdt:P18 ?img.
          ?item wdt:P21 wd:Q6581097.
          ?item wdt:P27 ?country.
          {?country wdt:P30 wd:Q46} UNION {?country wdt:P30 wd:Q49}



          FILTER (LANG(?label) = 'en') .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }.
        } LIMIT 600"""

    query = query % key


    results = get_results(endpoint_url, query)


    for result in results["results"]["bindings"]:

        url= result['img']['value']
        label = value + "/" + re.sub(" ", "_", result['label']['value']) + "_" + value + ".jpg"

        try:
            urllib.request.urlretrieve(url, label)
        except urllib.error.HTTPError:
            continue
