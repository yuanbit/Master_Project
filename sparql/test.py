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
# occupation: Q2159907 criminal
# occupation: Q43373553 cult leader
# occupation: Q484188 serial killer

# gender: Q6581097 Male

def get_results(endpoint_url, query):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

query = """SELECT distinct ?item ?label ?img
    WHERE{

          ?item wdt:P1399 ?crime.

          ?item rdfs:label ?label.

          ?item wdt:P18 ?img.
          ?item wdt:P21 wd:Q6581097.

          ?item wdt:P27 ?country.
          {?country wdt:P30 wd:Q46} UNION {?country wdt:P30 wd:Q49}


          FILTER (LANG(?label) = 'en') .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }.

        } LIMIT 500"""


results = get_results(endpoint_url, query)

for result in results["results"]["bindings"]:

    url= result['img']['value']
    label = "convict/" + re.sub(" ", "_", result['label']['value']) + "_convict.jpg"

    urllib.request.urlretrieve(url, label)
