
<!---
<div align="center">
    <img width="20%" alt="alpaca" src="https://user-images.githubusercontent.com/31032829/227085934-f4e7f99f-8b98-4c96-a091-e0f8743b6fb5.png">
</div>
--->

# :parrot: [ParroT](https://www.researchgate.net/publication/369797448_PARROT_Translating_During_Chat_Using_Large_Language_Models): Translating During Chat Using Large Language Models

:fire: News: Busy in organizing codes, data, and pretrained models. All are coming soon.
- :hugs: Try the pretrained models at HuggingFace model hub: [[Alpaca-7b]](https://huggingface.co/wxjiao/alpaca-7b), [[ParroT-7b]](https://huggingface.co/wxjiao/ParroT-7b), [[ParroT-Hint-7b]](https://huggingface.co/wxjiao/ParroT-Hint-7b)

> Parrots are smart birds that can respond to simple commands or questions. The question is whether they're just mimicking, or really intelligent enough to communicate with humans. This is similar to what we currently speculate about LLMs.

> Promoting the good is essential, but punishing the evil is also necessary to ensure that goodness prevails. Similarly, aligning LLMs with human feedbacks is exactly to learn from correct examples and discriminate erroneous examples.

Large language models (LLMs) like ChatGPT and GPT-4 have exhibited remarkable abilities on a wide range of natural language processing (NLP) tasks, including various machine translation abilities accomplished during chat. However, these models are only accessible through restricted APIs, which creates barriers to new research and advancements in the field. Therefore, we propose the ParroT framework to enhance and regulate the translation abilities during chat based on open-sourced LLMs (e.g., [LLaMA](https://github.com/facebookresearch/llama)) and human written translation and evaluation data. Specifically, ParroT reformulates translation data into the instruction-following style, and introduces a “Hint” field for incorporating extra requirements to regulate the translation process.

<div align="center">
    <img width="50%" alt="LLMs-MT" src="https://user-images.githubusercontent.com/31032829/230255125-bcf7393c-fd3c-4210-a3c6-60dc86a9721d.png">
    <p class="image-caption">Figure 1: Framework of ParroT. Hints are (optional) extra requirements to regulate the translation process.</p>
</div>


## Configurations

### Datasets

- Train Data: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), 
- Test Data: [Flores subsets](https://github.com/wxjiao/Is-ChatGPT-A-Good-Translator), [WMT22 test sets](https://www.statmt.org/wmt22/translation-task.html)
- Instruction-following format:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
We are translating the following sentences from Chinese to English.
    
### Input:
检查情况显示，市场销售的粮油、肉类、水果、蔬菜、蛋奶等生活必需品供应充足，商品价格基本稳定，未发现严重违法违规行为，市场经营秩序总体平稳。

### Hint: A translation with major accuracy/mistranslation errors could be

### Response:The results of the inspection indicate the sufficient supply of living necessities <v>on marketing</v> 
including cereals and oils, meat, fruits, vegetables, eggs and milk, and the basically stabilized commodity price. 
The inspection hasn’t found serious violation of laws and regulations. The market order is stable on an overall basis.
```

<!---
<div align="center">
    <img width="70%" alt="LLMs-MT" src="https://user-images.githubusercontent.com/31032829/227153636-fcaa0c4a-5bbd-4c78-9004-8ab988c71836.png">
    <p class="image-caption">Figure 0: Translation performance of LLMs on Flores subsets.</p>
</div>
--->

### Environment

We develop ParroT based on LLaMA with Hugging Face's transformers library by installing it from a particular fork (refer to this [PR](https://github.com/huggingface/transformers/pull/21955)). The hash of the specific commit we installed was `3884da12ce327667d4df5101aef3533cc32be61f`.

Framework Versions:
- Python 3.8.12
- Pytorch 1.10.0
- Transformers 4.27.0.dev0
```
# Clone the fork or use the transformers provided in this repo
git clone --branch llama_push  https://github.com/zphang/transformers.git

# Install
cd transformers
pip install .
```
- Datasets 2.10.0
- Tokenizers 0.13.2
- Others
```
pip install -r requirements.txt
```


### Finetune
TODO


### Inference (`inference.py`)

The script supports generation with and without hints using different instructions. 
The hints are appended to the default instruction with `###` as a delimiter.
Simply switch the inference instruction for different strategies. 

- None: instruct_inf.txt 
    - `Translate the following sentences from [SRC] to [TGT].`
- No Errors: instruct_inf_e2t.txt 
    - `Translate the following sentences from [SRC] to [TGT].###A translation with no errors could be`
- Minor Errors: instruct_inf_e2t_minor.txt 
    - `Translate the following sentences from [SRC] to [TGT].###A translation with minor errors could be`
- Major Errors: instruct_inf_e2t_major.txt 
    - `Translate the following sentences from [SRC] to [TGT].###A translation with major errors could be`
- Preferred: instruct_inf_t2t.txt 
    - `Translate the following sentences from [SRC] to [TGT].###We prefer to translate it to`

Example usage:
```
# Translation
python3 inference.py --model-name-or-path 'wxjiao/ParroT-Hint-7b' \
    -lp 'zh-en' \
    -t 0.1 \
    -sa 'beam' \
    -ins test/instruct_inf.txt \
    -i test/test_rand_50.zh.txt \
    -o test/test_rand_50.zh-en.none-hint.txt
    
# Text generation
python3 inference.py --model-name-or-path 'wxjiao/ParroT-Hint-7b' \
    -t 0.7 \
    -sa 'sample' \
    -i test/test_case.txt \
    -o test/test_case.general-task.txt
```


## Public Impact

### Citation
Please kindly cite our paper if you find it helpful:

```ruby
@inproceedings{jiao2023parrot,
  title={ParroT: Translating During Chat Using Large Language Models}, 
  author={Wenxiang Jiao and Jen-tse Huang and Wenxuan Wang and Xing Wang and Shuming Shi and Zhaopeng Tu},
  booktitle = {ArXiv},
  year      = {2023}
}
```
