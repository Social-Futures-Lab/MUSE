# MUSE

This repository provides the data and code to reproduce *[Correcting misinformation on social media with a large language model](https://arxiv.org/abs/2403.11169)*.

## Instructions

This repository contains three folders to reproduce (1) our proposed model, **MUSE**, as well as the results from (2) the expert evaluation and (3) the user study.

### Our Model, MUSE

We comply with X/Twitter Terms of Service by only releasing tweet IDs. To successfully run the model (i.e., `main.py`), you need to 
- Add your API keys in `data/api_keys.json`; 
- Obtain tweet content and image urls to populate `tweets.csv` and `image_url.json` based on our released tweet IDs.

### Expert Evaluation

- `data/notes_all.csv`: The [Community Notes](https://communitynotes.twitter.com/guide/en/about/introduction) data that our evaluation is based on.
- `data/responses.csv`: It contains the tweets and responses made by
  1. Laypeople with *high* helpfulness;
  2. Laypeople with *average* helpfulness; 
  3. MUSE that simulates correcting misinformation at the same time as the laypeople's *high*-helpfulness response;
  4. MUSE that simulates correcting misinformation at the same time as the laypeople's *average*-helpfulness response; 
  5. MUSE that simulates correcting misinformation *right after* it appears on social media;
  6. MUSE\retrieval (multimodal inputs only, otherwise it is the same as GPT-4);
  7. MUSE\vision (multimodal inputs only, otherwise it is the same as GPT-4); and
  8. GPT-4.

  '~' indicates the response is the same as (iii). '|||': the same as (iv). '*': the same as (vi). '$': the same as (vii). '///': the same as (viii). 
- `data/Q[..].csv`: It contains the experts' evaluation results of the responses in `data/responses.csv`.
- `data/username_tweetids.csv`: The assignment of the tweets and responses to every expert in the annotation phase.
- `data/tweetid_domain`: The identified domain of each tweet.
- `data/tweetid_misleadtype`: The identified tactic(s) of each tweet.
- `data/tweetid_politics`: The identified political learning of each tweet.
- `data/tweetid_factchecked`: The identified tweets that have been fact-checked online.
- `code/`: The code to reproduce the main results in our paper. The results were generated with Python 3.7 and dependencies in `requirements.txt`.
### Notes: 
- We comply with X/Twitter Terms of Service by only releasing the IDs of tweets. Most code files are runnable without further obtaining the tweet data, except `fig_s23.ipynb`, where the creation times of tweets are necessary. 
- The names of the experts are anonymized.

### User Study

- `data/posts.csv`: The tweet posts and responses used in the user study.
- `data/pre_belief.csv`: Users' ratings of whether tweets were misleading before reading responses (on a 7-point scale from "1: Extremely Accurate" to "7: Extremely Misleading").
- `data/post_belief.csv`: Users' ratings of whether tweets were misleading after reading responses (on the same 7-point scale).
- `data/pre_intention.csv`: Users' intentions to share the tweet before reading responses (on a 7-point scale from "1: Extremely Unlikely" to "7: Extremely Likely").
- `data/post_intention.csv`: Users' intentions to share the tweet after reading responses (on the same 7-point scale).
- `data/trustworthiness.csv`: Users' perceptions of the trustworthiness of the responses (on a 7-point scale from "1: Extremely Untrustworthy" to "7: Extremely Trustworthy").
- `code/analysis.ipynb`: The code to reproduce the user study results in our paper, including:
  1. Change in belief that misinformation is misleading before and after reading responses
  2. Change in intention to share misinformation before and after reading responses
  3. Trustworthiness of responses


## Citation
```
@article{zhou2024muse,
  title={Correcting misinformation on social media with a large language model},
  author={Zhou, Xinyi and Sharma, Ashish and Zhang, Amy X and Althoff, Tim},
  journal={arXiv preprint arXiv:2403.11169},
  year={2024}
}
```