
<!---
<div align="center">
    <img width="20%" alt="alpaca" src="https://user-images.githubusercontent.com/31032829/227085934-f4e7f99f-8b98-4c96-a091-e0f8743b6fb5.png">
</div>
--->

# :parrot: [ParroT](https://www.researchgate.net/publication/369797448_PARROT_Translating_During_Chat_Using_Large_Language_Models): Translating During Chat Using Large Language Models

:fire: News: Busy in organizing codes, data, and pretrained models. All are coming soon.
- :hugs: Try the pretrained models at HuggingFace model hub: [[Alpaca-7b]](https://huggingface.co/wxjiao/alpaca-7b) [[ParroT-7b]](https://huggingface.co/wxjiao/ParroT-7b)

Large language models (LLMs) like ChatGPT and GPT-4 have exhibited remarkable abilities on a wide range of natural language processing (NLP) tasks, including various machine translation abilities accomplished during chat. However, these models are only accessible through restricted APIs, which creates barriers to new research and advancements in the field. Therefore, we propose the ParroT framework to enhance and regulate the translation abilities during chat based on opensourced LLMs (e.g., [LLaMA](https://github.com/facebookresearch/llama)) and human written translation and evaluation data. Specifically, ParroT reformulates translation data into the instruction-following style, and introduces a “Hint” field for incorporating extra requirements to regulate the translation process. 

<div align="center">
    <img width="50%" alt="LLMs-MT" src="https://user-images.githubusercontent.com/31032829/230255125-bcf7393c-fd3c-4210-a3c6-60dc86a9721d.png">
    <p class="image-caption">Figure 0: Framework of ParroT. Hints are (optional) extra requirements to regulate the translation process.</p>
</div>



## Machine Translation

- **Data**: Flores subsets from [Is-ChatGPT-A-Good-Translator](https://github.com/wxjiao/Is-ChatGPT-A-Good-Translator)
- **Systems**: Google Translate, DeepL, ChatGPT, GPT-4, LLaMA, and Alpaca/Alpaca-MT (reproduced)
- **Environment**: Huggingface 4.27.0.dev0  <!-- (commit ID: 3884da1) -->
- **Prompt Format**: We follow Alpaca inference format.
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Translate the following sentences from Chinese to English.

他称，他制作了一个 WiFi 门铃。

### Response:He said that he had made a WiFi doorbell.
```

<div align="center">
    <img width="70%" alt="LLMs-MT" src="https://user-images.githubusercontent.com/31032829/227153636-fcaa0c4a-5bbd-4c78-9004-8ab988c71836.png">
    <p class="image-caption">Figure 0: Translation performance of LLMs on Flores subsets.</p>
</div>



**Results**:
Obviously, the vanilla LLaMA-7b model performs badly on all the four translation directions. By inspecting the outputs, we find that the vanilla LLaMA-7b model tends to generate very long sentences when translating (e.g., copy the instructions, continuing text expansion), which makes the generated text not faithful to the source sentences and also not grammatically correct. The reason could be the long context modeling during pretraining. Another reason is that we use the Alpaca inference format, which is basically a zero-shot setting that exhibits not guidance for translation. Besides, it is also not sure if the weight conversion by HuggingFace induces any information loss. 

Tuning LLaMA on the Alpaca dataset (52K) can ameliorate the above issue. Basically, Alpaca can produce complete translations that have similar lengths as the references. The translation performance is also boosted noticeably. However, Alpace still lags behind the commercial translation systems and ChatGPT/GPT-4 significantly.

Further, we combine the Alpaca dataset with an equally sized translation data (50K) and tune the LLaMA-7b model, resulting in Alpaca-MT model. This brings a significant improvement for all the translation directions, especically for En-Zh where Chinese data was hardly used during pretraining. 
