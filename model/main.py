import pandas as pd
import numpy as np
import json
import pprint
import string
import re
import time
from datetime import timedelta, datetime
import requests
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import AutoFeatureExtractor, AutoModel
import multiprocessing
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
from tqdm import tqdm
from newsplease import NewsPlease
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY)
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from transformers import pipeline
from huggingface_hub import InferenceClient
from openai import OpenAI

def llama(api_key, prompt):
    client = InferenceClient("meta-llama/Meta-Llama-3.1-70B-Instruct", token=api_key)
    response = []
    messages = [{"role": "user", "content": prompt}]
    stream = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        messages=messages,
        max_tokens=500,
        stream=True
    )
    for chunk in stream:
        response.append(chunk.choices[0].delta.content)
    return ''.join(response)


def gpt(api_key, prompt):
    max_retries = 20
    curr_tries = 1
    while curr_tries <= max_retries:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model='gpt-4',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                timeout=10
            )
            message = response.choices[0].message.content
            return message
        except Exception as e:
            print('\t'+str(e))
            curr_tries += 1
            time.sleep(5)
            continue
    return ''


def image2text(tweet, llm_key):
    def image2text_prep(tweet):
        def ocrs_prep(tweet, threshold=80):
            # Load OCR results
            with open('data/image_ocr.json', 'r') as f:
                aws_textract_api_outputs = json.load(f)

            image_ocrs = ['' for _ in range(len(tweet['tweet_images']))]
            for idx, image in enumerate(tweet['tweet_images']):
                image_output = aws_textract_api_outputs[image.split('/')[-1]]
                if str(type(image_output)) == '<class \'dict\'>' and 'TextDetections' in image_output.keys():
                    image_ocrs[idx] = " ".join([x['DetectedText'] for x in image_output['TextDetections']
                                                if 'Confidence' in x.keys() and x['Confidence'] > threshold and x['Type'] == 'LINE'])
            return image_ocrs

        def celebrities_prep(tweet):
            # Load celebrity recognition results
            with open('data/image_celebrity.json', 'r') as f:
                aws_rekognition_api_outputs = json.load(f)

            image_celebrities = ['' for _ in range(len(tweet['tweet_images']))]
            for idx, image in enumerate(tweet['tweet_images']):
                image_output = aws_rekognition_api_outputs[image.split('/')[-1]]
                if str(type(image_output)) == '<class \'dict\'>' \
                        and 'CelebrityFaces' in image_output.keys() \
                        and image_output['CelebrityFaces'] != []:
                    celeb_names = []
                    for cf in image_output['CelebrityFaces']:
                        celeb_names.append(cf['Name'])

                    image_celebrities[idx] = ', '.join(celeb_names)
            return image_celebrities

        def caps_prep(tweet):
            # Load captioning results
            with open('data/image_caption.json', 'r') as f:
                blip2_api_outputs = json.load(f)

            image_captions = ['' for _ in range(len(tweet['tweet_images']))]
            for idx, image in enumerate(tweet['tweet_images']):
                image_captions[idx] = blip2_api_outputs[image.split('/')[-1]]

            return image_captions

        def img_cap_clean(image_caption):
            social_media_list = ['youtube', 'facebook','instagram','pinterest','linkedin','snapchat','twitter','whatsapp','tiktok','reddit']
            words = image_caption.split()
            for idx, word in enumerate(words):
                if word in ["man", "woman"]:
                    words[idx] = "person"
                elif word in ["men", "women"]:
                    words[idx] = "people"
                if word.lower() in social_media_list:
                    words[idx] = "social media"

            return ' '.join(words)

        ocrs = ocrs_prep(tweet)
        celebrities = celebrities_prep(tweet)
        captions = [img_cap_clean(cap) for cap in caps_prep(tweet)]

        # Cross-examination
        prompt_formate_tweet = 'tweet text:\n[TWEET_TEXT]\n\nimage caption:\n[IMAGE_CAPTION]\n\nGiven {tweet text} and {image caption}, you are required to revise {image caption} based on the following rules:\n- Do **minimal** revision {image caption} based on {tweet text} if and only if when {tweet text} explicitly refutes {image caption}; otherwise, {revise image caption} should be the same as {image caption}.\n- Each person\'s name in {image caption} should also appear in {revised image caption}.\n- Do not add context from {tweet text} in {revised image caption} that is new to {image caption}.\n- {revised image caption} should follow the format of {image caption}.\n\nrevised image caption:'
        prompt_formate_ocr = 'image OCR:\n[IMAGE_OCR]\n\nimage caption:\n[IMAGE_CAPTION]\n\nGiven {image OCR} and {image caption}, you are required to revise {image caption} based on the following rules:\n- Do **minimal** revision {image caption} based on {image OCR} if and only if when {image OCR} explicitly refutes {image caption}; otherwise, {revise image caption} should be the same as {image caption}.\n- Do not add context from {image OCR} in {revised image caption} that is new to {image caption}.\n- {revised image caption} should follow the format of {image caption}.\n\nrevised image caption:'

        for idx in range(len(tweet['tweet_images'])):
            if celebrities[idx] != '':
                temp = []
                for celebrity in celebrities[idx].split(', '):
                    if celebrity.lower() in tweet['tweet_text'].lower() or celebrity.lower() in ocrs[idx].lower():
                        temp.append(celebrity)
                celebrities[idx] = ', '.join(temp)

            if captions[idx] != '':
                prompt = prompt_formate_tweet.replace('[TWEET_TEXT]', tweet['tweet_text'])
                prompt = prompt.replace('[IMAGE_CAPTION]', str(captions[idx]))
                captions[idx] = gpt(llm_key, prompt)

                if ocrs[idx] != '':
                    prompt = prompt_formate_ocr.replace('[IMAGE_OCR]', ocrs[idx])
                    prompt = prompt.replace('[IMAGE_CAPTION]', str(captions[idx]))
                    captions[idx] = gpt(llm_key, prompt)

        return ocrs, celebrities, captions

    ocrs, celebrities, captions = image2text_prep(tweet)

    with open('data/prompt_informative_image_captioning.txt', 'r') as f:
        prompt_formate = f.read()

    tweet['tweet_image2text'] = ['' for _ in range(len(tweet['tweet_images']))]
    for idx in range(len(tweet['tweet_images'])):
        if captions[idx] == '':
            continue

        if ocrs[idx] == '' and celebrities[idx] == '':
            tweet['tweet_image2text'][idx] = 'A photo of ' + captions[idx] + '.'
        else:
            prompt = prompt_formate.replace('[CAPTION]', captions[idx])
            prompt = prompt.replace('[OCR]', ocrs[idx])
            prompt = prompt.replace('[CELEBRITIES]', celebrities[idx])

            tweet['tweet_image2text'][idx] = gpt(llm_key, prompt)
            print('\t(sent 1 request to llm)')
            if tweet['tweet_image2text'][idx][0] == '{' and tweet['tweet_image2text'][idx][-1] == '}':
                tweet['tweet_image2text'][idx] = tweet['tweet_image2text'][idx][1:-1]

    return tweet


def query_generation(tweet, llm_key):
    # Prepare tweet data
    def um_tweet_prep(tweet_text, tweet_user, tweet_time):
        if "yesterday" in tweet_text.lower():
            yesterdate = str(tweet_time - timedelta(days=1)).split(' ')[0]

            insensitive_yesterday = re.compile(re.escape("yesterday"), re.IGNORECASE)
            tweet_text = insensitive_yesterday.sub('on ' + yesterdate, tweet_text)

        insensitive_tweet_text = tweet_text.translate(str.maketrans('', '', string.punctuation))
        insensitive_words = insensitive_tweet_text.lower().split()
        if "i" in insensitive_words and "breaking" in insensitive_words:
            todate = str(tweet_time - timedelta(days=0)).split(' ')[0]
            return tweet_text + ' (Tweeted by ' + tweet_user + ' on ' + todate + ')'
        elif "i" in insensitive_words:
            return tweet_text + ' (Tweeted by ' + tweet_user + ')'
        elif "breaking" in insensitive_words:
            todate = str(tweet_time - timedelta(days=0)).split(' ')[0]
            return tweet_text + ' (Tweeted on ' + todate + ')'
        else:
            return tweet_text

    def mm_tweet_prep(tweet_text, tweet_img2text, tweet_user, tweet_time):
        if "yesterday" in tweet_text.lower():
            yesterdate = str(tweet_time - timedelta(days=1)).split(' ')[0]

            insensitive_yesterday = re.compile(re.escape("yesterday"), re.IGNORECASE)
            tweet_text = insensitive_yesterday.sub('on ' + yesterdate, tweet_text)

        insensitive_tweet_text = tweet_text.translate(str.maketrans('', '', string.punctuation))
        insensitive_words = insensitive_tweet_text.lower().split()
        if "i" in insensitive_words and "breaking" in insensitive_words:
            todate = str(tweet_time - timedelta(days=0)).split(' ')[0]
            return tweet_text + ' (Tweeted by ' + tweet_user + ' on ' + todate + '. Attached images: ' + tweet_img2text + ')'
        elif "i" in insensitive_words:
            return tweet_text + ' (Tweeted by ' + tweet_user + '. Attached images: ' + tweet_img2text + ')'
        elif "breaking" in insensitive_words:
            todate = str(tweet_time - timedelta(days=0)).split(' ')[0]
            return tweet_text + ' (Tweeted on ' + todate + '. Attached images: ' + tweet_img2text + ')'
        else:
            return tweet_text + ' (Attached images: ' + tweet_img2text + ')'

    post_content = ''
    if tweet['tweet_modality'] == 'unimodal':
        post_content = um_tweet_prep(p.clean(tweet['tweet_text']),
                                     tweet['user_name'],
                                     datetime.strptime(tweet['created_time'], '%Y-%m-%d %H:%M:%S'))
        # Load the prompt template
        with open('data/prompt_query_generation_unimodal.txt', 'r') as f:
            prompt_format = f.read()
    elif tweet['tweet_modality'] == 'multimodal':
        post_content = mm_tweet_prep(p.clean(tweet['tweet_text']),
                                     ' '.join(tweet['tweet_image2text']),
                                     tweet['user_name'],
                                     datetime.strptime(tweet['created_time'], '%Y-%m-%d %H:%M:%S'))
        # Load the prompt template
        with open('data/prompt_query_generation_multimodal.txt', 'r') as f:
            prompt_format = f.read()
    # Prepare the prompt
    prompt = prompt_format.replace('[POST_CONTENT]', post_content)
    # Generate queries with GPT-4
    response = gpt(llm_key, prompt)

    return response


def multimodal_search(api_key, tweet, page_num):
    def serpapi_google_reverse_img_search(api_key, image_url, query, start_index):
        results = {}
        params = {
            "api_key": api_key,
            "engine": "google_reverse_image",
            "google_domain": "google.com",
            "hl": "en",
            "lr": "lang_en",
            "no_cache": True,
            "image_url": image_url,
            "start": start_index,
            "q": query
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
        except Exception as e:
            print('\t(fail to get ' + image_url + ')')
        return results

    search_results = {}
    search_results['page_' + str(page_num+1)] = {}

    for image_url in tweet['tweet_images']:
        if image_url == '':
            continue

        queries = tweet['queries']
        if queries in ['NONE', '\"NONE\"']: # No queries, search using images
            save_file = 'eval/data/web_search/reverse_image/' + image_url.split('/')[-1] + '_page' + str(page_num+1) + '.json'
            if not os.path.exists(save_file):
                multimodal_search_outputs = serpapi_google_reverse_img_search(api_key, image_url, '', page_num * 10)
                print('\t(sent 1 request to serpapi)')
                with open(save_file, 'w') as f:
                    json.dump(multimodal_search_outputs, f, indent=4)
            else:
                with open(save_file, 'r') as f:
                    multimodal_search_outputs = json.load(f)
            search_results['page_' + str(page_num+1)][image_url] = multimodal_search_outputs
        else:
            queries = queries.split('\n')
            queries = [q for q in queries if len(q) > 0 and q[0] in ['1', '2', '3', '4', '5']]
            # print('---------query---------')
            for x, query_x in enumerate(queries):
                qx = ' '.join(query_x.split(' ')[1:])
                qx = qx.translate(str.maketrans('', '', string.punctuation))
                # print(qx)
                save_file = 'data/web_search_multimodal/' + image_url.split('/')[-1] + '_' + qx + '_page' + str(page_num+1) + '.json'
                if not os.path.exists(save_file):
                    multimodal_search_outputs = serpapi_google_reverse_img_search(api_key, image_url, qx, page_num * 10)
                    print("\t(sent 1 request to serpapi)")
                    with open(save_file, 'w') as f:
                        json.dump(multimodal_search_outputs, f, indent=4)
                else:
                    with open(save_file, 'r') as f:
                        multimodal_search_outputs = json.load(f)
                search_results['page_' + str(page_num+1)][image_url + '_' + qx] = multimodal_search_outputs

    return search_results


def reverse_image_search(api_key, tweet, page_num=0):
    def serpapi_google_reverse_img_search(api_key, image_url, query, start_index):
        results = {}
        params = {
            "api_key": api_key,
            "engine": "google_reverse_image",
            "google_domain": "google.com",
            "hl": "en",
            "lr": "lang_en",
            "no_cache": True,
            "image_url": image_url,
            "start": start_index,
            "q": query
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
        except Exception as e:
            print('\t(fail to get ' + image_url + ')')
        return results

    search_results = {}

    for image_url in tweet['tweet_images']:
        if image_url == '':
            continue

        save_file = 'data/web_search_multimodal/' + image_url.split('/')[-1] + '_page' + str(page_num+1) + '.json'
        if not os.path.exists(save_file):
            multimodal_search_outputs = serpapi_google_reverse_img_search(api_key, image_url, '', page_num * 10)
            print('\t(sent 1 request to serpapi)')
            with open(save_file, 'w') as f:
                json.dump(multimodal_search_outputs, f, indent=4)
        else:
            with open(save_file, 'r') as f:
                multimodal_search_outputs = json.load(f)
        search_results[image_url] = multimodal_search_outputs

    return search_results


def query_search(api_key, search_engine_id, tweet, domain_priority):
    def google_programmable_search(api_key, search_engine_id, query):
        params = {'key': api_key, 'cx': search_engine_id, 'q': query}
        search = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        results = search.json()
        return results

    search_results = {}
    queries = tweet["queries"]
    # No queries, return {}
    if queries in ["NONE", "\"NONE\""]:
        return search_results
    # Otherwise, [query_1, query_2, ...]
    queries = queries.split('\n')
    queries = [q for q in queries if len(q)>0 and q[0] in ['1', '2', '3']]
    # print('---------query---------')
    for x, query_x in enumerate(queries):
        qx = ' '.join(query_x.split(' ')[1:])
        # print(qx)
        qx = qx.translate(str.maketrans('', '', string.punctuation))
        save_file = 'data/web_search_text/'+qx+'_'+domain_priority+'.json'
        # If the search results have not been saved, do web search. Otherwise, load the search results
        if not os.path.exists(save_file):
            with open(save_file, 'w') as f:
                json.dump({}, f, indent=4)
        with open(save_file, 'r') as f:
            google_programmable_search_outputs = json.load(f)
        if 'items' in google_programmable_search_outputs.keys():
            search_results[qx] = google_programmable_search_outputs
        else:
            output = google_programmable_search(api_key, search_engine_id, qx)
            print("\t(sent 1 request to google programmable search)")
            if 'items' in output.keys():
                search_results[qx] = output
            google_programmable_search_outputs = output
            with open(save_file, 'w') as f:
                json.dump(google_programmable_search_outputs, f, indent=4)
    return search_results


def filter_search_results_by_domain(search_results, domain_priority, page_num=-1):
    with open('data/publisher_priority_' + domain_priority.lower() + '.txt', 'r') as f:
        selected_domains = f.read().split('\n')

    if page_num == -1:
        sr = search_results
    else:
        sr = search_results['page_' + str(page_num+1)]

    filtered_search_links = set()
    for results in sr.values():
        if 'image_results' not in results.keys():
            continue
        for res in results['image_results']:
            domain = res['link'].split('/')[2]
            domain = domain.replace('www.', '')
            if domain in selected_domains:
                filtered_search_links.add(res['link'])

    return filtered_search_links


def filter_search_results_by_query_sim(search_results, top_k):
    filtered_search_links = set()
    for results in search_results.values():
        for i in range(min(top_k, len(results['items']))):
            filtered_search_links.add(results['items'][i]['link'])
    return list(filtered_search_links)


def article_crawler(article_urls, driver):

    searched_articles = []

    articles_json = 'data/web_page_content.json'
    if not os.path.exists(articles_json):
        with open(articles_json, 'w') as f:
            json.dump({}, f)
    with open(articles_json, 'r') as f:
        saved_articles = json.load(f)

    for _, article_url in tqdm(enumerate(article_urls)):
        # Extract the content
        if article_url in saved_articles.keys():
            article = saved_articles[article_url]
        else:
            try:
                driver.get(article_url)
            except Exception as e:
                print('\t(fail to get ' + article_url + ')')
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            article = NewsPlease.from_html(soup.prettify(), article_url).get_dict()

            saved_articles[article_url] = article
            with open(articles_json, 'w') as f:
                json.dump(saved_articles, f, sort_keys=True, default=str, indent=4)

        # Process the content
        if article['maintext'] and len(article['maintext'].split()) >= 100 and article['date_publish'] and article['language'] == 'en':
            # Remove the duplicate content
            flag = 0
            for searched_article in searched_articles:
                curr_article_time = datetime.strptime(str(article['date_publish']), '%Y-%m-%d %H:%M:%S')
                ref_article_time = datetime.strptime(str(searched_article['date_publish']), '%Y-%m-%d %H:%M:%S')
                if article['maintext'] in searched_article['maintext'] and (curr_article_time - ref_article_time).total_seconds() >= 0:
                    flag = 1
                    break
                elif searched_article["maintext"] in article["maintext"] and (ref_article_time - curr_article_time).total_seconds() >= 0:
                    searched_articles.remove(searched_article)
                    searched_articles.append(article)
                    flag = 1
                    break
            if flag == 0:
                searched_articles.append(article)

    return searched_articles


def filter_search_results_by_time(tweet, searched_articles):
    filtered_searched_articles = []

    time_threshold = datetime.strptime(tweet['created_time'], '%Y-%m-%d %H:%M:%S')
    for article in searched_articles:
        time_article = datetime.strptime(str(article['date_publish']), '%Y-%m-%d %H:%M:%S')
        if (time_threshold - time_article).total_seconds() > 0:
            filtered_searched_articles.append(article)

    return filtered_searched_articles


def compute_multimodal_sims(tweet, article):

    def get_img_sim(model_name, img_url_1, img_url_2):
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        hidden_dim = model.config.hidden_size

        def extract_embeddings(image):
            image_pp = extractor(image, return_tensors='pt')
            features = model(**image_pp).last_hidden_state[:, 0].detach().numpy()
            return features.squeeze()

        try:
            img_bytes_1 = extract_embeddings(Image.open(BytesIO(requests.get(img_url_1, timeout=5).content)).convert('RGB'))
            img_bytes_2 = extract_embeddings(Image.open(BytesIO(requests.get(img_url_2, timeout=5).content)).convert('RGB'))

            cos_sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            return cos_sim(img_bytes_1, img_bytes_2)

        except:
            return -1

    def get_text_sim(model, text_1, text_2):
        embeds_1 = model.encode(text_1, convert_to_tensor=True).cpu()
        embeds_2 = model.encode(text_2, convert_to_tensor=True).cpu()
        return util.dot_score(embeds_1, embeds_2).numpy()[0][0]

    articleUrl_mmSims = {article["url"]: {'text': float(-1), 'image': float(-1)}}

    # Prepare tweet text and images (# images can be > 1 but <= 4)
    image2text_comb = '\n'.join([i2t for i2t in tweet['tweet_image2text'] if i2t != ''])
    if image2text_comb == '':
        tweet_content = p.clean(tweet['tweet_text'])
    else:
        tweet_content = p.clean(tweet['tweet_text']) + '\n' + image2text_comb

    tweet_images = tweet['tweet_images']

    # Prepare article text and image (#=1)
    article_content = article['maintext']
    article_image = article['image_url']

    text_sim = get_text_sim(SentenceTransformer('msmarco-distilbert-base-tas-b'), tweet_content, article_content)
    articleUrl_mmSims[article['url']]['text'] = text_sim

    img_sims = []
    for tweet_image in tweet_images:
        img_sims.append(get_img_sim('facebook/dino-vitb8', tweet_image, article_image))
    articleUrl_mmSims[article["url"]]['image'] = sum(img_sims) / len(img_sims)

    return articleUrl_mmSims


def filter_search_results_by_multimodal_sim(list_of_articleUrl_mmSims, searched_articles, threshold_text=95, threshold_image=0.7):
    filtered_searched_links = set()

    article_urls = [article_url for articleUrl_mmSims in list_of_articleUrl_mmSims for article_url in
                    articleUrl_mmSims.keys()]
    text_sims = [mm_sims['text'] for articleUrl_mmSims in list_of_articleUrl_mmSims for mm_sims in
                 articleUrl_mmSims.values()]
    image_sims = [mm_sims['image'] for articleUrl_mmSims in list_of_articleUrl_mmSims for mm_sims in
                  articleUrl_mmSims.values()]

    max_index = np.argmax(np.array(text_sims))
    if text_sims[max_index] >= threshold_text:
        filtered_searched_links.add(article_urls[max_index])

    max_index = np.argmax(np.array(image_sims))
    if image_sims[max_index] >= threshold_image:
        filtered_searched_links.add(article_urls[max_index])

    filtered_searched_articles = []
    for article in searched_articles:
        if article['url'] in filtered_searched_links:
            filtered_searched_articles.append(article)

    return filtered_searched_articles


def compute_unimodal_sims(tweet, article):
    def get_text_sim(model, text_1, text_2):
        embeds_1 = model.encode(text_1, convert_to_tensor=True).cpu()
        embeds_2 = model.encode(text_2, convert_to_tensor=True).cpu()
        return util.dot_score(embeds_1, embeds_2).numpy()[0][0]

    articleUrl_textSim = {article['url']: float(-1)}

    text_sim = get_text_sim(SentenceTransformer('msmarco-distilbert-base-tas-b'),
                            p.clean(tweet['tweet_text']), article['maintext'])
    articleUrl_textSim[article['url']] = text_sim

    return articleUrl_textSim


def filter_search_results_by_unimodal_sim(list_of_articleUrl_textSim, searched_articles, max_num=3, threshold=90):
    filtered_searched_links = set()

    article_urls = [article_url for articleUrl_textSim in list_of_articleUrl_textSim for article_url in
                    articleUrl_textSim.keys()]
    text_sims = [text_sim for articleUrl_textSim in list_of_articleUrl_textSim for text_sim in
                 articleUrl_textSim.values()]

    textSims_articleUrls = list(zip(text_sims, article_urls))
    textSims_articleUrls.sort(reverse=True)

    article_urls = [article_url for _, article_url in textSims_articleUrls]
    text_sims = [text_sim for text_sim, _ in textSims_articleUrls]

    x = min(len(text_sims), max_num)
    for idx in range(x):
        if text_sims[idx] >= threshold:
            filtered_searched_links.add(article_urls[idx])

    filtered_searched_articles = []
    for article in searched_articles:
        if article['url'] in filtered_searched_links:
            filtered_searched_articles.append(article)

    return filtered_searched_articles


def evidence_extraction(tweet, article, llm_key):
    output = ([], [])

    tweet_text = p.clean(tweet['tweet_text'])
    tweet_user = tweet['user_name']
    tweet_time = tweet['created_time']
    tweet_user_id = '@' + tweet['user_screen_name']
    tweet_user_description = tweet['user_description']

    article_content = article['maintext']
    article_time = str(article['date_publish'])
    article_url = article['url']

    prompt = ''
    if tweet['tweet_modality'] == 'multimodal':

        tweet_img2text = ' '.join(tweet['tweet_image2text'])

        # Use the first max_chars characters
        article_content = article_content[:20000]

        with open('data/prompt_evidence_extraction.txt', 'r') as f:
            prompt_format = f.read()
        prompt = prompt_format.replace('[ARTICLE_CONTENT]', article_content)

        if article_time != '':
            prompt = prompt.replace('[ARTICLE_PUBLISH_DATE]', ' (published on ' + article_time + ')')
        else:
            prompt = prompt.replace('[ARTICLE_PUBLISH_DATE]', '')

        if tweet_user_description != '':
            prompt = prompt.replace('[TWEET_CONTENT]', tweet_text +
                                    '\n(Tweeted by ' + tweet_user + '. User ID: ' + tweet_user_id + '. User description: {' + tweet_user_description + '})' +
                                    '\n(Tweeted on ' + tweet_time + ')' +
                                    '\n(Attached images: {' + tweet_img2text + '})')
        else:
            prompt = prompt.replace('[TWEET_CONTENT]', tweet_text +
                                    '\n(Tweeted by ' + tweet_user + '. User ID: ' + tweet_user_id + ')' +
                                    '\n(Tweeted on ' + tweet_time + ')' +
                                    '\n(Attached images: {' + tweet_img2text + '})')

    elif tweet['tweet_modality'] == 'unimodal':
        # Use the first max_chars characters
        article_content = article_content[:20000]

        with open('data/prompt_evidence_extraction.txt', 'r') as f:
            prompt_format = f.read()
        prompt = prompt_format.replace('[ARTICLE_CONTENT]', article_content)

        if article_time != '':
            prompt = prompt.replace('[ARTICLE_PUBLISH_DATE]', ' (published on ' + article_time + ')')
        else:
            prompt = prompt.replace('[ARTICLE_PUBLISH_DATE]', '')

        if tweet_user_description != '':
            prompt = prompt.replace('[TWEET_CONTENT]', tweet_text +
                                    '\n(Tweeted by ' + tweet_user + '. User ID: ' + tweet_user_id + '. User description: {' + tweet_user_description + '})' +
                                    '\n(Tweeted on ' + tweet_time + ")")
        else:
            prompt = prompt.replace('[TWEET_CONTENT]', tweet_text +
                                    '\n(Tweeted by ' + tweet_user + '. User ID: ' + tweet_user_id + ')' +
                                    '\n(Tweeted on ' + tweet_time + ')')

    evidence = gpt(llm_key, prompt)
    print("\t(sent 1 request to llm)")

    if 'explicit refutation' in evidence.lower() and 'implicit refutation' in evidence.lower():
        evidences = evidence.split('\n')
        for i, evidence in enumerate(evidences):
            if 'explicit refutation' in evidence.lower():
                er_loc = i
            elif 'implicit refutation' in evidence.lower():
                ce_loc = i
        refute_evidence, context_evidence = evidences[er_loc+1:ce_loc], evidences[ce_loc+1:]

        refute_evidence_2 = []
        for i, s in enumerate(refute_evidence):
            if 'NONE' in s or len(s)==0 or (len(s)>0 and s[0]!='-'):
                continue
            if article_time != '':
                subs = ' Reference: ' + article_url + ' published on ' + article_time
            else:
                subs = ' Reference: ' + article_url
            if len(s)>1 and s[2] == "\"" and s[-1] == "\"":
                refute_evidence_2.append('- ' + s[3:-1] + subs)
            else:
                refute_evidence_2.append(s + subs)
        # print('---------refute_evidence---------')
        # print(refute_evidence_2)

        context_evidence_2 = []
        for i, s in enumerate(context_evidence):
            if 'NONE' in s or len(s)==0 or (len(s)>0 and s[0]!='-'):
                continue
            if article_time != '':
                subs = ' Reference: ' + article_url + ' published on ' + article_time
            else:
                subs = ' Reference: ' + article_url
            if len(s)>1 and s[2] == "\"" and s[-1] == "\"":
                context_evidence_2.append('- ' + s[3:-1] + subs)
            else:
                context_evidence_2.append(s + subs)
        # print('---------context_evidence---------')
        # print(context_evidence_2)

        output = (refute_evidence_2, context_evidence_2)

    return output


def correction_generation(tweet, list_of_evidences, list_of_refute_evidences, llm_key):
    if len(list_of_refute_evidences) == 0:
        with open('data/prompt_correction_generation_without_retrieval.txt', 'r') as f:
            prompt = f.read()
    else:
        with open('data/prompt_correction_generation_with_retrieval.txt', 'r') as f:
            prompt = f.read()
        prompt = prompt.replace('[RELEVANT_FACTS]', '\n'.join(list_of_evidences))

    tweet_text = p.clean(tweet['tweet_text'])
    tweet_user = tweet['user_name']
    tweet_time = tweet['created_time']
    tweet_user_id = '@' + tweet['user_screen_name']
    tweet_user_description = tweet['user_description']

    if tweet['tweet_modality'] == 'multimodal':
        tweet_img2text = ' '.join(tweet['tweet_image2text'])

        if tweet_user_description != '':
            prompt = prompt.replace('[TWEET_CONTENT]', tweet_text +
                                    '\n(Tweeted by ' + tweet_user + '. User ID: ' + tweet_user_id + '. User description: {' + tweet_user_description + '})' +
                                    '\n(Tweeted on ' + tweet_time + ')' +
                                    '\n(Attached images: {' + tweet_img2text + '})')
        else:
            prompt = prompt.replace('[TWEET_CONTENT]', tweet_text +
                                    '\n(Tweeted by ' + tweet_user + '. User ID: ' + tweet_user_id + ')' +
                                    '\n(Tweeted on ' + tweet_time + ')' +
                                    '\n(Attached images: {' + tweet_img2text + '})')

    elif tweet['tweet_modality'] == 'unimodal':
        if tweet_user_description != '':
            prompt = prompt.replace('[TWEET_CONTENT]', tweet_text +
                                    '\n(Tweeted by ' + tweet_user + '. User ID: ' + tweet_user_id + '. User description: {' + tweet_user_description + '})' +
                                    '\n(Tweeted on ' + tweet_time + ')')
        else:
            prompt = prompt.replace('[TWEET_CONTENT]', tweet_text +
                                    '\n(Tweeted by ' + tweet_user + '. User ID: ' + tweet_user_id + ')' +
                                    '\n(Tweeted on ' + tweet_time + ')')

    return gpt(llm_key, prompt)


def get_article_publish_date(article):
    return datetime.strptime(str(article['date_publish']), "%Y-%m-%d %H:%M:%S")


if __name__ == '__main__':
    TEST_TWEET_ID = '1621717259449257984'
    INPUT_DATA_FILE = 'data/tweets_unimodal.csv'

    data = pd.read_csv(INPUT_DATA_FILE, dtype=str)
    data = data.fillna('')

    instance = data[data['tweet_id'] == TEST_TWEET_ID].to_dict('records')[0]
    instance['tweet_modality'] = INPUT_DATA_FILE.split('/')[1].split('.')[0].split('_')[1]

    # Load API keys
    with open('data/api_keys.json', 'r') as f:
        api_keys = json.load(f)
    llm_key = api_keys['OpenAI']
    # llm_key = api_keys['HuggingFace']
    serpapi_key = api_keys['SerpAPI']
    gsearch_key = api_keys['GoogleSearch']['Key']

    # Open a browser for article crawling
    options = Options()
    driver = webdriver.Firefox(options=options)
    driver.implicitly_wait(5)
    driver.set_page_load_timeout(10)

    # Set the number of processes = 5
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=5)

    #### When misinformation is unimodal ####
    if instance['tweet_modality'] == 'unimodal':

        print('Start correcting unimodal misinformation...')

        print('Generate queries from misinformation...')
        instance['queries'] = query_generation(instance, llm_key)

        refute_evidences, context_evidences = set(), set()
        for domain_priority in ['High', 'Medium', 'Low']:
            print('Search web pages with ' + domain_priority + ' priority...')
            search_results = query_search(gsearch_key, api_keys['GoogleSearch'][domain_priority+'-Priority'], instance, domain_priority)

            print('Start selecting retrieved web pages...')

            # Filtered based on their similarities with the query
            filtered_search_links = filter_search_results_by_query_sim(search_results, 3)

            print('Extract retrieved web page content...')
            searched_articles = article_crawler(filtered_search_links, driver)
            if searched_articles == []:
                print('\t(no article found)')
                continue

            # Filtered based on their publication times
            # Used to simulate correcting misinfo immediately after its appearance
            filtered_searched_articles = filter_search_results_by_time(instance, searched_articles)
            if filtered_searched_articles == []:
                print('\t(no article qualified based on time)')
                continue

            # Filtered based on their similarities with the misinfo content
            print('Compute the similarity between web page and misinformation content...')
            params = [(instance, article) for article in filtered_searched_articles]
            list_of_articleUrl_textSim = pool.starmap(compute_unimodal_sims, params)
            final_searched_articles = filter_search_results_by_unimodal_sim(list_of_articleUrl_textSim, filtered_searched_articles)
            if final_searched_articles == []:
                print('\t(no article qualified based on similarity)')
                continue

            print('Extract evidence from selected web pages...')
            params = [(instance, article, llm_key) for article in final_searched_articles]
            evidences = pool.starmap(evidence_extraction, params)

            for refute_evidence, context_evidence in evidences:
                refute_evidences = refute_evidences.union(set(refute_evidence))
                context_evidences = context_evidences.union(set(context_evidence))

            if len(refute_evidences) >= 3:
                print('\t(stop with sufficient refutations)')
                break

        if len(refute_evidences) == 0:
            print('\nExtend web search due to no refutations...')

            refute_evidences, context_evidences = set(), set()
            for domain_priority in ['High', 'Medium', 'Low']:
                print('Search web pages with ' + domain_priority + ' priority...')
                search_results = query_search(gsearch_key, api_keys['GoogleSearch'][domain_priority + '-Priority'],
                                              instance, domain_priority)

                print('Start selecting retrieved web pages...')

                # Filtered based on their similarities with the query
                filtered_search_links = filter_search_results_by_query_sim(search_results, 10)

                print('Extract retrieved web page content...')
                searched_articles = article_crawler(filtered_search_links, driver)
                if searched_articles == []:
                    print('\t(no article found)')
                    continue

                # Filtered based on their publication times
                # Used to simulate correcting misinfo immediately after its appearance
                filtered_searched_articles = filter_search_results_by_time(instance, searched_articles)
                if filtered_searched_articles == []:
                    print('\t(no article qualified based on time)')
                    continue

                # Filtered based on their similarities with the misinfo content
                print('Compute the similarity between web page and misinformation content...')
                params = [(instance, article) for article in filtered_searched_articles]
                list_of_articleUrl_textSim = pool.starmap(compute_unimodal_sims, params)
                final_searched_articles = filter_search_results_by_unimodal_sim(list_of_articleUrl_textSim,
                                                                                filtered_searched_articles)
                if final_searched_articles == []:
                    print('\t(no article qualified based on similarity)')
                    continue

                print('Extract evidence from selected web pages...')
                params = [(instance, article, llm_key) for article in final_searched_articles]
                evidences = pool.starmap(evidence_extraction, params)

                for refute_evidence, context_evidence in evidences:
                    refute_evidences = refute_evidences.union(set(refute_evidence))
                    context_evidences = context_evidences.union(set(context_evidence))

                if len(refute_evidences) >= 3:
                    print('\t(stop with sufficient refutations)')
                    break

        refute_evidences = list(refute_evidences)
        context_evidences = list(context_evidences)

        instance['refute_evidence'] = refute_evidences
        instance['context_evidence'] = context_evidences

        evidences = refute_evidences + context_evidences

        print('Generate correction with ' + str(len(refute_evidences)) + ' refutations...')
        instance['correction'] = correction_generation(instance, evidences, refute_evidences, llm_key)

    #### When misinformation is multimodal ####
    elif instance['tweet_modality'] == 'multimodal':

        print('Start correcting multimodal misinformation...')

        # Process images
        instance['tweet_images'] = []
        with open('data/image_url.json', 'r') as f:
            imgId_imgUrl = json.load(f)
        for img_id, img_url in imgId_imgUrl.items():
            if instance["tweet_id"] == img_id.split('_')[0] and img_url != '':
                instance["tweet_images"].append(img_url)

        print('Generate textual description of images...')
        instance = image2text(instance, llm_key)

        print('Generate queries from misinformation...')
        instance['queries'] = query_generation(instance, llm_key)

        refute_evidences, context_evidences = set(), set()
        for domain_priority in ['High', 'Medium', 'Low']:
            print('Search web pages with ' + domain_priority + ' priority...')
            search_results = {}
            filtered_search_links = set()
            for page_num in range(5):
                print('\t(page ' + str(page_num+1) + ')')
                search_results.update(multimodal_search(serpapi_key, instance, page_num))
                if search_results['page_' + str(page_num+1)] == {}:
                    # print(search_results)
                    print("\t(stop with no web pages)")
                    break
                filtered_search_links = filtered_search_links.union(
                    filter_search_results_by_domain(search_results, domain_priority, page_num))
                if len(filtered_search_links) > 10:
                    print("\t(stop with sufficient web pages)")
                    break
            filtered_search_links = list(filtered_search_links)
            # for l in filtered_search_links:
            #     print(l)

            print('Start selecting retrieved web pages...')

            print('Extract retrieved web page content...')
            searched_articles = article_crawler(filtered_search_links, driver)
            if searched_articles == []:
                print('\t(no article found)')
                continue

            # Filtered based on their publication times
            # Used to simulate correcting misinfo immediately after its appearance
            filtered_searched_articles = filter_search_results_by_time(instance, searched_articles)
            if filtered_searched_articles == []:
                print('\t(no article qualified based on time)')
                continue

            # Filtered based on their similarities with the misinfo content
            print('Compute the similarity between web page and misinformation content...')
            params = [(instance, article) for article in filtered_searched_articles]
            list_of_articleUrl_mmSims = pool.starmap(compute_multimodal_sims, params)

            final_searched_articles = filter_search_results_by_multimodal_sim(list_of_articleUrl_mmSims, filtered_searched_articles)
            if final_searched_articles == []:
                print("\t(no article qualified based on similarity)")
                continue

            print('Extract evidence from selected web pages...')
            params = [(instance, article, llm_key) for article in final_searched_articles]
            evidences = pool.starmap(evidence_extraction, params)

            for refute_evidence, context_evidence in evidences:
                refute_evidences = refute_evidences.union(set(refute_evidence))
                context_evidences = context_evidences.union(set(context_evidence))

            if len(refute_evidences) >= 3:
                print("\t(stop with sufficient refutations)")
                break

        if len(refute_evidences) < 3:
            print('\nConduct unimodal web search due to insufficient refutations...')

            #### Unimodal search, search filter, and extract evidence ###
            for domain_priority in ['High', 'Medium', 'Low']:
                print('Search web pages with ' + domain_priority + ' priority...')
                # Search by text or images
                image_search_results = reverse_image_search(serpapi_key, instance, 0)
                text_search_results = query_search(gsearch_key, api_keys['GoogleSearch'][domain_priority+'-Priority'], instance, domain_priority)

                print('Start selecting retrieved web pages...')
                filtered_image_search_links = filter_search_results_by_domain(image_search_results, domain_priority)
                filtered_text_search_links = filter_search_results_by_query_sim(text_search_results, 3)
                filtered_search_links = list(set(filtered_text_search_links).union(filtered_image_search_links))
                # for l in filtered_search_links:
                #     print(l)

                print('Extract retrieved web page content...')
                searched_articles = article_crawler(filtered_search_links, driver)
                if searched_articles == []:
                    print('\t(no article found)')
                    continue

                # Filtered based on their publication times
                # Used to simulate correcting misinfo immediately after its appearance
                filtered_searched_articles = filter_search_results_by_time(instance, searched_articles)
                if filtered_searched_articles == []:
                    print('\t(no article qualified based on time)')
                    continue

                # Filtered based on their similarities with the misinfo content
                print('Compute the similarity between web page and misinformation content...')
                params = [(instance, article) for article in filtered_searched_articles]
                list_of_articleUrl_mmSims = pool.starmap(compute_multimodal_sims, params)

                final_searched_articles = filter_search_results_by_multimodal_sim(list_of_articleUrl_mmSims, filtered_searched_articles)
                if final_searched_articles == []:
                    print("\t(no article qualified based on similarity)")
                    continue

                print('Extract evidence from selected web pages...')
                params = [(instance, article, llm_key) for article in final_searched_articles]
                evidences = pool.starmap(evidence_extraction, params)

                for refute_evidence, context_evidence in evidences:
                    refute_evidences = refute_evidences.union(set(refute_evidence))
                    context_evidences = context_evidences.union(set(context_evidence))

                if len(refute_evidences) >= 3:
                    print("\t(stop with sufficient refutations)")
                    break

        refute_evidences = list(refute_evidences)
        context_evidences = list(context_evidences)

        instance["refute_evidences"] = refute_evidences
        instance["context_evidences"] = context_evidences

        evidences = refute_evidences + context_evidences

        print('Generate correction with ' + str(len(refute_evidences)) + ' refutations...')
        instance['correction'] = correction_generation(instance, evidences, refute_evidences, llm_key)

    with open('data/output/' + instance["tweet_id"] + ".json", 'w') as f:
        json.dump(instance, f, indent=4, sort_keys=True)

    print()
    print(instance['tweet_id'])
    print(instance['correction'])

    driver.quit()
