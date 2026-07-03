import pandas as pd
import json
import os
from tqdm import tqdm
import time
from time import sleep
import requests

from urllib.parse import urlparse
import sys
from googleapiclient.discovery import build

import trafilatura
from trafilatura.meta import reset_caches
from trafilatura.settings import DEFAULT_CONFIG
import pymupdf

from timeout_decorator import timeout
from functools import lru_cache

DEFAULT_CONFIG.MAX_FILE_SIZE = 50000

### IMPORT FILES
# Load API keys from file
if os.path.isdir('/config.json'):
  with open('/config.json', 'r') as f:
    config = json.load(f)
  SEARCH_KEY = config['search_key']
  SEARCH_ENGINE_ID = config['search_engine_id']

### GATHER KNOWLEDGE STORE URLS
# Helper Functions
def call_search_api(search_term, **kwargs):
  """Executes a single call to the Google Search API.

  Parameters
  ----------
  search_term : string
          The text to be used as a search query.
  kwargs
          All other arguments for the cse().list function.

  Returns
  ----------
  out : list
          A list of Response objects. If no results exist, returns an empty list.
  """
  # Code adapted from Schlichtkrull (2024)
  service = build("customsearch", "v1", developerKey=SEARCH_KEY)
  res = service.cse().list(q=search_term, cx=SEARCH_ENGINE_ID, **kwargs).execute()
  time.sleep(0.1) # Sleep, to avoid rate limits

  if "items" in res:
    return res['items'] # Return list of 'Results' objects
  else:
    return []

# Calls Google API on a search string, with given parameters.
# Adapted from Schlichtkrull (2024)
def get_google_search_results(search_string, sort_date=None, page=0):
  """Retrieves a page of search results for the given query.
  Performs error handling if search fails.

  Each page of results (usually) contains 10 entries.

  Parameters
  ----------
  search_string : string
          The text to be used as a search query
  sort_date : string
          If given, only sources before this date will be returned.
          to be given in format YYYYMMDD.
  page : int
          The page number of results to return.
          Note that page=0 returns the first page of results.

  Returns
  ----------
  out : list
          A list of Response objects. If search fails, returns an empty list.
  """
  search_results = []
  sort = None if sort_date is None else ("date:r:19000101:"+sort_date)

  # Try 3 times to search
  for _ in range(3):
    try:
      search_results += call_search_api(
        search_string,
        num=10,
        start=10*page + 1,
        sort=sort,
        dateRestrict=None,
        gl="US"
      )
      break
    except:
      print("I encountered an error trying to search \""+search_string+"\". Maybe the connection dropped. Trying again in 3 seconds...", file=sys.stderr)
      time.sleep(3)
  return search_results

# Define search queries; ‹source› will be replaced by source name
SEARCH_QUERIES = [
    '"‹source›" news about',
    '"‹source›" news ownership',
    '"‹source›" news endorsement',
    '"‹source›" news bias',
    '"‹source›" news failed fact checks',
    ]

def create_store_for_source(source_name, num_retrieved_pages=5):
  """Retrieves multiple pages (10 entries per page) per search query for the given source.
  Retrieves 5 pages by default.

  Parameters
  ----------
  source_name : string
          The name of the source to perform queries for. All extraneous endings
          not part of the source name (e.g. .txt endings) should be stripped.
  num_retrieved_pages : int
          The number of pages to retrieve.

  Returns
  ----------
  out : dict
          A dictionary of search queries as keys and a corresponding list of Response objects
          as values.
  """
  out_dict = {}
  for query in SEARCH_QUERIES:
    specific_query = query.replace('‹source›', source_name)

    # Get search results for each page
    query_results = []
    for page in range(num_retrieved_pages):
      query_results.extend(get_google_search_results(specific_query, page=page))

    # Save all pages of query search results to dict
    out_dict[specific_query] = query_results
  return out_dict

def build_knowledge_store(sources_file_name):
  """Gather search results over 5 queries via the Google Search API,
  for each source in the given file.

  Parameters
  ----------
  sources_file_name : string
          The filepath for the sources list document

  Returns
  ----------
  out : dict
          A dictionary of search queries as keys and a corresponding list of Response objects
          as values.
  """
  # Load, preprocess source names
  with open(sources_file_name, 'r') as f:
    list_sources = [line.strip().removesuffix('.txt') for line in f]

  full_results = {}
  iter_sources = tqdm(list_sources)
  for source_name in iter_sources:
    iter_sources.set_postfix({'current source' : source_name})

    # Retrieve 10 pages of search results for each query; append to overall store
    source_results = create_store_for_source(source_name, num_retrieved_pages=10)
    full_results |= source_results

  # Write dict to file, with retrieved sources associated to queries
  out_file_name = sources_file_name.removesuffix('.tsv')
  with open(f'{out_file_name}_full_results.json', 'w') as f:
    f.write(json.dumps(full_results, indent=2))

  # Combine retrieved sources into single list, without associated queries,
  # and write to file
  full_results_list = []
  for val in full_results.values():
    full_results_list += val
  df_combined = pd.DataFrame(full_results_list)
  df_combined.to_json(f'{out_file_name}_combined.json',
                      # orient='records',
                      # lines=True,
                      indent=2
                      )

  return df_combined

# Build knowledge store
df_store = build_knowledge_store('/data/mbc_list.tsv')

# Build actual store in parts, to avoid overall usage limits
# df_store_half1 = build_knowledge_store('mbc_list_half1.tsv')
# df_store_half2 = build_knowledge_store('mbc_list_half2.tsv')

# Load gold urls
with open('gold_urls.tsv', 'r') as f:
  gold_links = f.read().split('\n')

# Remove links to MB/FC
gold_links = [link for link in gold_links if 'mediabiasfactcheck.com' not in link]

# Deduplicate URLs
df_store = df_store.drop_duplicates(subset='link')
gold_links_dedup = []
for link in gold_links:
  if link not in gold_links_dedup:
    gold_links_dedup.append(link)

# Append gold links to knowledge store if not already there
addl_links = []
for link in gold_links_dedup:
  if link not in df_store.link.values:
    addl_links.append({
        'kind' : 'gold_url',
        'link' : link,
        'snippet' : '(No snippet available)'
    })
df_addl_links = pd.DataFrame(addl_links)
df_extended_store = pd.concat([df_store, df_addl_links]).reset_index(drop=True)

# Save final set of links for knowledge store
df_extended_store.to_json('knowledge_store_urls.json')

### WEB SCRAPING
# Blacklist of files not to web-scrape (From Schlichtkrull 2024)
blacklist = [
  "jstor.org", # Blacklisted because their pdfs are not labelled as such, and clog up the download
  "facebook.com", # Blacklisted because only post titles can be scraped, but the scraper doesn't know this,
  "ftp.cs.princeton.edu", # Blacklisted because it hosts many large NLP corpora that keep showing up
  "nlp.cs.princeton.edu",
  "mediabiasfactcheck.com", # We don't want to retrieve the test data
  "ground.news" # Cites mediabiasfactcheck too often
]
blacklist_files = [ # Blacklisted some additional NLP files that show up in search results and cause OOM errors
  "/glove.",
  "ftp://ftp.cs.princeton.edu/pub/cs226/autocomplete/words-333333.txt",
  "https://web.mit.edu/adamrose/Public/googlelist"
]

# Helper functions for trafilatura (from Schlichtkrull 2024)
@timeout(10)
def __get_url__(url):
    return trafilatura.fetch_url(url, config=DEFAULT_CONFIG)

# Try up to max no. of attempts to retrieve page
max_tries = 2
@lru_cache(maxsize=2048)
def get_page(url):
    page = None
    for i in range(max_tries):
        try:
            page = __get_url__(url)
            assert page is not None
            #print("Fetched "+url, file=sys.stderr)
            break
        except:
            #print("Failed to fetch "+url, file=sys.stderr)

            if i < max_tries - 1:
                sleep(1)
                #print("Trying again.", file=sys.stderr)
    return page

def url2text(url):
    page = get_page(url)
    if page is None:
        return ""
    lines = html2text(page)
    return lines

def html2text(page):
    out_lines = ""

    if len(page.strip()) == 0 or page is None:
        return out_lines

    text = trafilatura.extract(page, config=DEFAULT_CONFIG)
    reset_caches()

    if text is None:
        return out_lines
    return text

def url_to_filename(url):
  return url.replace('://', '_').replace('.', '_').replace('/', '_').removesuffix('/')

def get_domain_name(url):
  if '://' not in url:
    url = 'http://' + url

  domain = urlparse(url).netloc
  if domain.startswith("www."):
    return domain[4:]
  else:
    return domain

# Given URL, check if it is scrapeable. If so, perform full web-scraping process
# and return text.
# (Adapted from Schlichtkrull 2024)
def retrieve_text(url, process_pdfs=False):
  domain = get_domain_name(url)

  # Terminate on blacklisted sources, unscrapeable file types
  if domain in blacklist:
    return None
  if any(b_file in url for b_file in blacklist_files):
    return None
  if url.endswith(".doc"):
    return None

  # Scrape pdf text, if enabled
  if url.endswith(".pdf"):
    if process_pdfs:
      r = requests.get(url)
      data = r.content
      doc = pymupdf.Document(stream=data)
      text = '\n'.join([page.get_text() for page in doc])
    else:
      return None
  else:
    text = url2text(url)

  # Skip invalid lines in text
  keep_lines = []
  for line in text.split("\n"):
    # Keep lines with 3 or more words
    if len(line.strip().split(" ")) < 3:
      continue
    if line in ["Something went wrong. Wait a moment and try again.",
                "Please enable Javascript and refresh the page to continue"]:
      continue

    keep_lines.append(line.strip())

  out_text = '\n'.join(keep_lines)
  return out_text

# Make output dir
out_dir = '/store_texts'
if not os.path.exists(out_dir):
  os.makedirs(out_dir)

# Iterate through links to retrieve from
store_links = tqdm(df_extended_store.link)
for link in store_links:
  store_links.set_postfix_str(link+' ')

  # Convert URL to valid filename
  out_filename = url_to_filename(link)
  if len(out_filename) > 200:
    out_filename = out_filename[:200]

  # If file already exists in output dir, skip
  if os.path.exists(os.path.join(out_dir, out_filename)):
    # print(f'Skipping already-retrieved url {link}')
    continue

  # Scrape and save to folder
  text = retrieve_text(link)
  if text is None:
    continue
  with open(os.path.join(out_dir, out_filename), 'w') as f:
    f.write(text)