import re
import pandas as pd
import json
import os
from tqdm import tqdm

# Load API keys from file
if os.path.isdir('/config.json'):
  with open('/config.json', 'r') as f:
    config = json.load(f)
  SEARCH_KEY = config['search_key']
  SEARCH_ENGINE_ID = config['search_engine_id']

# Given text document, extract raw url path and display text
def extract_urls(text):
  # Define patterns for matching URLs and section headings separately
  url_pattern = r'(\[.*?\])(\(http.*?\))'
  section_pattern = r'(history\n|funded by / ownership\n|analysis / bias\n|analysis\n|bias\n|failed fact checks\n)'

  # Return and process all matches.
  # Matches are of form ('display_text', 'url_text', 'section_heading')
  match_pattern = rf'{url_pattern}|{section_pattern}'
  url_list = re.findall(match_pattern, text, re.IGNORECASE)
  return process_urls(url_list)

def process_urls(url_list):
  # Given list of urls, remove those we do not want to analyse.
  blacklist_phrases = [
    'read our profile',
    'view our country profile',
  ]

  cleaned_urls = []
  current_sect = ''
  for display_text, url_text, section_head in url_list:
    # Update current section heading
    if section_head:
      current_sect = section_head.strip()
      continue

    # Skip blacklisted urls
    if any(b_phrase in display_text.lower() for b_phrase in blacklist_phrases):
      continue

    # Strip opening, closing brackets from display text and url
    cleaned_urls.append((display_text[1:-1], url_text[1:-1], current_sect.title()))
  return cleaned_urls

# Set filepath of gold mbc texts to extract URLs from
mbc_dirpath = '/data/gold_mbcs'
mbc_namelist = os.listdir(mbc_dirpath)

# Extract URLs for gold MBCs
list_mbc_url_data = []
for source_name in tqdm(mbc_namelist):
  mbc_filepath = os.path.join(mbc_dirpath, source_name)
  with open(mbc_filepath, 'r') as f_mbc:
    mbc_text = f_mbc.read()
    mbc_urls = extract_urls(mbc_text)

    # count_url_types = Counter((url_type for _, _, url_type in mbc_urls))
    list_mbc_url_data.append({
        'Source' : source_name[:-4],
        'Wordcount' : len(mbc_text.split()),
        'URLs' : mbc_urls,
        'URLcount' : len(mbc_urls),
        # 'URL_typecount' : count_url_types,
    })
df_urls = pd.DataFrame(list_mbc_url_data)

with open('/data/mbc_list.tsv', 'r') as f:
  source_list = [line.strip().removesuffix('.txt') for line in f]

url_list = []
for source in source_list:
  source_url_list = df_urls.loc[df_urls.Source==source, 'URLs'].values[0]
  source_url_list = [link for _, link, _ in source_url_list]
  url_list.extend(source_url_list)

# Save retrieved URLs to file
with open('gold_urls.tsv', 'w') as f:
  f.write('\n'.join(url_list))