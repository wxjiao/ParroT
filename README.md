
<!---
<div align="center">
    <img width="20%" alt="alpaca" src="https://user-images.githubusercontent.com/31032829/227085934-f4e7f99f-8b98-4c96-a091-e0f8743b6fb5.png">
</div>
--->

# :parrot: [ParroT](https://www.researchgate.net/publication/369797448_PARROT_Translating_During_Chat_Using_Large_Language_Models): Translating During Chat Using Large Language Models

:fire: News: Busy in organizing codes, data, and pretrained models. All are coming soon.
- :hugs: Try the pretrained models at HuggingFace model hub: [[Alpaca-7b]](https://huggingface.co/wxjiao/alpaca-7b), [[ParroT-7b]](https://huggingface.co/wxjiao/ParroT-7b), [[ParroT-Hint-7b]](https://huggingface.co/wxjiao/ParroT-Hint-7b)

Large language models (LLMs) like ChatGPT and GPT-4 have exhibited remarkable abilities on a wide range of natural language processing (NLP) tasks, including various machine translation abilities accomplished during chat. However, these models are only accessible through restricted APIs, which creates barriers to new research and advancements in the field. Therefore, we propose the ParroT framework to enhance and regulate the translation abilities during chat based on opensourced LLMs (e.g., [LLaMA](https://github.com/facebookresearch/llama)) and human written translation and evaluation data. Specifically, ParroT reformulates translation data into the instruction-following style, and introduces a “Hint” field for incorporating extra requirements to regulate the translation process.

> Parrots are smart birds that can respond to simple commands or questions. The question is whether they're just mimicking, or really intelligent enough to communicate with humans. This is similar to what we currently speculate about LLMs.

> Promoting the good is essential, but punishing the evil is also necessary to ensure that goodness prevails. Similarly, aligning LLMs with human feedbacks is exactly to learn from correct examples and discriminate erroneous examples.


<div align="center">
    <img width="50%" alt="LLMs-MT" src="https://user-images.githubusercontent.com/31032829/230255125-bcf7393c-fd3c-4210-a3c6-60dc86a9721d.png">
    <p class="image-caption">Figure 1: Framework of ParroT. Hints are (optional) extra requirements to regulate the translation process.</p>
</div>


<!---
## Machine Translation

- **Data**: Flores subsets from [Is-ChatGPT-A-Good-Translator](https://github.com/wxjiao/Is-ChatGPT-A-Good-Translator)
- **Systems**: Google Translate, DeepL, ChatGPT, GPT-4, LLaMA, and Alpaca/Alpaca-MT (reproduced)
- **Environment**: Huggingface 4.27.0.dev0  <!-- (commit ID: 3884da1) -->
- **Prompt Format**: We follow Alpaca inference format.
--->


## Configurations

### Datasets

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Translate the following sentences from Chinese to English.

他称，他制作了一个 WiFi 门铃。

### Response:He said that he had made a WiFi doorbell.
```

<!---
<div align="center">
    <img width="70%" alt="LLMs-MT" src="https://user-images.githubusercontent.com/31032829/227153636-fcaa0c4a-5bbd-4c78-9004-8ab988c71836.png">
    <p class="image-caption">Figure 0: Translation performance of LLMs on Flores subsets.</p>
</div>
--->

### Environment

### Finetune

### Evaluate


## Public Impact

### Citation
Please kindly cite our report if you find it helpful:

```ruby
@inproceedings{jiao2023parrot,
  title={ParroT: Translating During Chat Using Large Language Models}, 
  author={Wenxiang Jiao and Jen-tse Huang and Wenxuan Wang and Xing Wang and Shuming Shi and Zhaopeng Tu},
  booktitle = {ArXiv},
  year      = {2023}
}
```
