# pip install sparqlwrapper
# https://rdflib.github.io/sparqlwrapper/

######### Download images of various professions from Wikidata ###############
from SPARQLWrapper import SPARQLWrapper, JSON
import re
import urllib.request

endpoint_url = "https://query.wikidata.org/sparql"

def get_results(endpoint_url, query):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

# The entity IDs are the keys and the professions are the values
occupations = {"Q189290":"military_officer", \
               "Q2462658": "manager", "Q41583": "coach", \
               "Q40348": "lawyer", "Q639669": "musician", \
               "Q33999": "actor", "Q33999": "entrepreneur", \
               "Q82955": "politician", "Q42973": "architect"}

# For all entity IDs in dictionary execute query
for key, value in occupations.items():
    # gender: Q6581097 Male
    query = """SELECT distinct ?name ?img
    WHERE{

          ?person wdt:P106 wd:%s.
          ?person rdfs:label ?name.
          ?person wdt:P18 ?img.
          ?person wdt:P21 wd:Q6581097.
          ?person wdt:P27 ?country.
          {?country wdt:P30 wd:Q46} UNION {?country wdt:P30 wd:Q49}
          FILTER (LANG(?name) = 'en') .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }.
        } LIMIT 2"""

    query = query % key
    results = get_results(endpoint_url, query)

    # Automatically download the images from the links
    for result in results["results"]["bindings"]:

        url= result['img']['value']
        # The directories of the profesions have already been created
        label = value + "/" + re.sub(" ", "_", result['name']['value']) + "_" + value + ".jpg"

        try:
            urllib.request.urlretrieve(url, label)
        except urllib.error.HTTPError:
            continue
