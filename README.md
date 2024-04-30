# MUSE

This repository provides the data and code to reproduce the results of *[Correcting misinformation on social media with a large language model](https://arxiv.org/abs/2403.11169)*. 

## Instructions
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
- `code/`: The code to reproduce the main results in our paper. The results were generated with Python 3.7 and dependencies in `requirements.txt`.
### Notes: 
- We comply with X/Twitter Terms of Service by only releasing the IDs of tweets. Most code files are runnable without further obtaining the tweet data, except `fig_s17.ipynb`, `fig_s27.ipynb`, and `fig_s28.ipynb`, where the creation times of tweets are necessary. 
- The names of the experts are anonymized.

## Citation
```
@article{zhou2024muse,
  title={Correcting misinformation on social media with a large language model},
  author={Zhou, Xinyi and Sharma, Ashish and Zhang, Amy X and Althoff, Tim},
  journal={arXiv preprint arXiv:2403.11169},
  year={2024}
}
```
