# Alpaca-MT
Investigating the capability of [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)-style [LLaMA](https://github.com/facebookresearch/llama) models for machine translation.


## Machine Translation

- **Data**: Flores subsets from [Is-ChatGPT-A-Good-Translator](https://github.com/wxjiao/Is-ChatGPT-A-Good-Translator)
- **Systems**: Google Translate, DeepL, ChatGPT, GPT-4, LLaMA, and Alpaca (reproduced)
- **Environment**: Huggingface v4.27.0 (commit ID: 3884da1)
- **Prompt Format**: We follow Alpaca inference format.
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Translate the following sentences from Chinese to English.

他称，他制作了一个 WiFi 门铃。

### Response:He said that he had made a WiFi doorbell.
```

<div align="center">
    <img width="70%" alt="LLMs-MT" src="https://user-images.githubusercontent.com/31032829/226847369-0251755e-5777-46c9-afb2-cde4b3a5ab73.png">
    <p class="image-caption">Figure 0: Translation performance of LLMs on Flores subsets.</p>
</div>

**Results**:
Obviously, the vanilla LLaMA-7b model performs badly on all the four translation directions. By inspecting the outputs, we find that the vanilla LLaMA-7b model tends to generate very long sentences when translating (e.g., copy the instructions, continuing text expansion), which makes the generated text not faithful to the source sentences and also not grammatically correct. The reason could be the long context modeling during pretraining. Another reason is that we use the Alpaca inference format, which is basically a zero-shot setting that exhibits not guidance for translation. Besides, it is also not sure if the weight conversion by HuggingFace induces any information loss. 

Tuning LLaMA on the Alpaca dataset can ameliorate the above issue. Basically, Alpaca can produce complete translations that have similar lengths as the references. The translation performance is also boosted noticeably. However, Alpace still lags behind the commercial translation systems and ChatGPT/GPT-4 significantly.




