

<div align="center">
    <img width="25%" alt="ParroT" src="https://github.com/wxjiao/ParroT/assets/31032829/9893aba1-7ea3-4c76-a995-9b12aff44950">
    <h2>
    ParroT: Translating During Chat Using Large Language Models <br><br>
     <a href="https://arxiv.org/abs/2304.02426"> <img alt="paper link" src="https://img.shields.io/badge/Paper-arXiv-red"> </a>
     <a href="https://github.com/wxjiao/InstructMT"> <img alt="data link" src="https://img.shields.io/badge/Data-InstructMT-blue"> </a> 
    </h2>
</div>

<!---
:parrot: 
# ParroT: Translating During Chat Using Large Language Models
--->

:fire: **Update**
- [2023/07/14] Incorporated [`flash-attention`](https://github.com/wxjiao/ParroT/blob/master/transformers/examples/pytorch/language-modeling/run_clm_llms_flash.py) into BLOOM for long-context training; observed about 20-30% speedup with other settings fixed.

<details> 

- [2023/06/14] Releasing detailed instruction data and scripts on [@InstructMT](https://github.com/wxjiao/InstructMT). 
- The WMT22 test sets are made available.
- For medium-to-small models (e.g., 7B), we recommend ZeRO2+offload rather than ZerO3; use gradient accumulation to maximize GPU usage.
- Important optimizations: `preprocess_function` to be 4-5X faster; `DataCollatorForSeq2Seq` for batch-wise padding to save  5-10% GPU usage.
- Introducing ParroT-LoRA which supports saving and restarting from the checkpoints (base model and lora weights) during finetuning.
- Setting the default Transformers to `>= 4.28.0.dev0` directly as it merged the PR of LLaMA. With this version on Torch 1.13.1 + CUDA 11.7, we find the finetuning process could be a bit faster (~18%) than our [v1.0.0](https://github.com/wxjiao/ParroT/tree/v1.0.0/transformers/examples/pytorch/language-modeling) implementation.

</details>

:star: **Highlight** :star:
- :hugs: Try the pretrained models at HuggingFace model hub:
  -  [[Alpaca-7b]](https://huggingface.co/wxjiao/alpaca-7b), [[ParroT-7b]](https://huggingface.co/wxjiao/ParroT-7b), [[ParroT-Hint-7b]](https://huggingface.co/wxjiao/ParroT-Hint-7b)
  -  [[ParroT-Hint-7b-lora]](https://huggingface.co/wxjiao/ParroT-Hint-7b-lora) based on [[LLaMA-7b]](https://huggingface.co/wxjiao/llama-7b)

<!---
- :page_facing_up: The preprint is available now on arxiv, refer to it for more details: [[paper]](https://arxiv.org/abs/2304.02426) 
--->


## ParroT

> Parrots are smart birds that can respond to simple commands or questions. The question is whether they're just mimicking, or really intelligent enough to communicate with humans. This is similar to what we currently speculate about LLMs.

> Promoting the good is essential, but punishing the evil is also necessary to ensure that goodness prevails. Similarly, aligning LLMs with human feedbacks is exactly to learn from correct examples and discriminate erroneous examples.

Large language models (LLMs) like ChatGPT and GPT-4 have exhibited remarkable abilities on a wide range of natural language processing (NLP) tasks, including various machine translation abilities accomplished during chat. However, these models are only accessible through restricted APIs, which creates barriers to new research and advancements in the field. Therefore, we propose the **ParroT** framework to enhance and regulate the translation abilities during chat based on open-sourced LLMs (e.g., [LLaMA](https://github.com/facebookresearch/llama), [Bloomz](https://huggingface.co/bigscience/bloomz)) and human written translation and evaluation data. Specifically, ParroT reformulates translation data into the instruction-following style, and introduces a “Hint” field for incorporating extra requirements to regulate the translation process.

<div align="center">
    <img width="60%" alt="LLMs-MT" src="https://github.com/wxjiao/ParroT/assets/31032829/bc791aa5-1c79-4ad7-bbee-f361a3b3009a">
    <p class="image-caption">Figure 1: Framework of ParroT. Hints are (optional) extra requirements to regulate the translation process.</p>
</div>


## Configurations

### Datasets

- Train Data: data/data_alpaca_hf.json, [data_parrot_hf.json](https://drive.google.com/file/d/1pQmj-eFwHycSkQtuAB3OKF47bHPxDVon/view?usp=share_link)
    - You can also use [Alpaca data by GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM): data/data_alpaca_gpt4_hf_en.json, data/data_alpaca_gpt4_hf_zh.json 
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



### Environment

We develop ParroT based on open-sourced LLMs (e.g., LLaMA, Bloomz) with HuggingFace's transformers library.

Framework Versions:
- Python 3.8.12
- Pytorch 1.13.1+cu117
- Transformers (git+https://github.com/huggingface/transformers.git) 
- Peft (git+https://github.com/huggingface/peft.git)
- Flash-attn
- Triton 2.0.0.dev20221202
- Other requirements
```
pip install -r requirements.txt
```


### Data Format Conversion

Convert the regular bilingual sentence pairs into Alpaca data format:
```
python3 scripts/convert_pair_to_alpaca.py \
    -s zh -t en \
    -if scripts/instruct_follow.txt \
    -sf data/train.zh-en.zh.txt \
    -tf data/train.zh-en.en.txt \
    -of data/train_alp.json
```

Convert the Alpaca data format to the training data format here:
```
python3 scripts/convert_alpaca_to_hf.py \
    -i data/train_alp.json \
    -o data/train_alp_hf.json
```


### Finetune
We modify the example script of language modeling in transformers for finetuning, i.e., `run_clm.py` with the built in HuggingFace `Trainer`.
So it would be easy to get started if you are familiar with `run_clm.py`. Also, this script supports data streaming, which might be helpful for handling larger datasets. [DeepSpeed ZeRO stage 2/3](https://github.com/microsoft/DeepSpeed) is adopted for distributed training.

The resulting finetuning scripts are named as [`run_clm_llms.py`](https://github.com/wxjiao/ParroT/blob/master/transformers/examples/pytorch/language-modeling/run_clm_llms.py) and [`run_clm_lora.py`](https://github.com/wxjiao/ParroT/blob/master/transformers/examples/pytorch/language-modeling/run_clm_lora.py) for full model training and LoRA training, respectively.
Theoretically, the `run_clm_lora.py` script can handle both full model and LoRA by specifying the arguments. But we also keep the former one for full model in consideration of safe development.

**For LoRA training, we recommend to use ZeRO2 since ZeRO3 is very unstable when saving `adapter_model.bin`.**

For long-context training, we provide the [`run_clm_llms_flash.py`](https://github.com/wxjiao/ParroT/blob/master/transformers/examples/pytorch/language-modeling/run_clm_llms_flash.py) to improve the memory efficiency.

LLaMA-7b:
- Original weights for the LLaMA models can be obtained by filling out this [Form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)
- Convert the LLaMA weights into the HuggingFace format by following the instructions in this [Doc](https://huggingface.co/docs/transformers/main/model_doc/llama)
- Optionally converted one [[LLaMA-7b]](https://huggingface.co/wxjiao/llama-7b)

Bloomz-7b1-mt:
- Available on HuggingFace: [Bloomz-7b1-mt](https://huggingface.co/bigscience/bloomz-7b1-mt)

Example usages on 8 A100 by 1 node:

<details>
<summary><b> Full Model </b></summary>

```
# Multi-nodes are also supported

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29500}"

train_path=transformers/examples/pytorch/language-modeling/run_clm_llms.py
model_path=<your_proj_path>/llama-7b
model_save=<your_proj_path>/parrot-hint-7b

# HOST_NUM will be 1
torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node 8 \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${train_path} \
    --deepspeed train/deepspeed_config_zero2.json \
    --model_name_or_path ${model_path} \
    --train_file data/data_parrot_hf.json \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1.5 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --block_size 512 \
    --do_train \
    --evaluation_strategy "no" \
    --validation_split_percentage 0 \
    --fp16 True \
    --fp16_full_eval True \
    --streaming \
    --ddp_timeout 3600 \
    --seed 1 \
    --gradient_checkpointing True \
    --output_dir ${model_save}
```
</details>


<details>
<summary><b> LoRA </b></summary>
    
```
# Multi-nodes are also supported

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29500}"

train_path=transformers/examples/pytorch/language-modeling/run_clm_lora.py
model_path=<your_proj_path>/llama-7b
model_save=<your_proj_path>/parrot-hint-lora-7b

# HOST_NUM will be 1
torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node 8 \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${train_path} \
    --deepspeed train/deepspeed_config_zero2.json \
    --model_name_or_path ${model_path} \
    --train_file data/data_parrot_hf.json \
    --use_lora True \
    --lora_config train/lora_config.json \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1.5 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --block_size 512 \
    --do_train \
    --evaluation_strategy "no" \
    --validation_split_percentage 0 \
    --fp16 True \
    --fp16_full_eval True \
    --streaming \
    --ddp_timeout 3600 \
    --seed 1 \
    --gradient_checkpointing True \
    --output_dir ${model_save}
```
    
</details>


### Inference

The scripts support generation with and without hints using different instructions. 
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

Example usages:

<details>
<summary><b> Full Model </b></summary>

```
# Translation
python3 inference.py --model-name-or-path <your_proj_path>/parrot-hint-7b \
    -lp 'zh-en' \
    -t 0.1 \
    -sa 'beam' \
    -ins test/instruct_inf.txt \
    -i test/test_rand_50.zh.txt \
    -o test/test_rand_50.zh-en.none-hint.txt
    
# Text generation
python3 inference.py --model-name-or-path <your_proj_path>/parrot-hint-7b \
    -t 0.7 \
    -sa 'sample' \
    -i test/test_case.txt \
    -o test/test_case.general-task.txt
```

</details>


<details>
<summary><b> LoRA </b></summary>

```
# Translation
python3 inference_lora.py --model-name-or-path <your_proj_path>/llama-7b \
    --lora-weights <your_proj_path>/parrot-hint-lora-7b/adapter_model \
    -lp 'zh-en' \
    -t 0.1 \
    -sa 'beam' \
    -ins test/instruct_inf.txt \
    -i test/test_rand_50.zh.txt \
    -o test/test_rand_50.zh-en.none-hint.txt
    
# Text generation
python3 inference_lora.py --model-name-or-path <your_proj_path>/llama-7b \
    --lora-weights <your_proj_path>/parrot-hint-lora-7b/adapter_model \
    -t 0.7 \
    -sa 'sample' \
    -i test/test_case.txt \
    -o test/test_case.general-task.txt
```

</details>


### MT Evaluation
We adopt two metrics, SacreBLEU and COMET (Unbabel/wmt22-comet-da), which are driven by _n_-gram similarity and cross-lingual pretrained models, respectively. 
```
# SacreBLEU
cat test_rand_50.zh-en.none-hint.txt.hyp | sacrebleu -w 2 test_rand_50.en.txt

# COMET
comet-score -r test_rand_50.en.txt -s test_rand_50.zh.txt -t test_rand_50.zh-en.none-hint.txt.hyp --quiet --only_system
```


### Finetuned LLMs and Results

Currently, we finetuned the following LLMs for ParroT with the evaluation mainly on WMT22 test sets.

- [x] LLaMA-7b
- [x] Bloomz-mt-7b
- [x] ParroT-LoRA
- [ ] 8bit Training (high requirements for both environments and GPU types)

There are several interesting observations:
- ParroT based on Bloomz-mt-7b also works well with hints. Besides, Bloomz-mt-7b shows stronger ability in the modeling of Chinese texts.
- LoRA seems to prevent LLMs from overfitting which benefits the high-resource De-En translation but restricts the instruction learning of other directions. The limited trainable parameters (only ~4.2M) may explain this observation.

<div align="center">
    <img width="70%" alt="alpaca" src="https://github.com/wxjiao/ParroT/assets/31032829/654014cc-5450-4855-bb05-aeaf9fd5b5da">
    <p class="image-caption">Caption: Translation performance of LLMs on Flores subsets and WMT22 test sets.</p>
</div>


## Run LLMs on your MacBook

Try [llama.cpp](https://github.com/ggerganov/llama.cpp) to run the LLMs using 4-bit quantization on a MacBook.
We adopt a specific fork from [comex/llama.cpp](https://github.com/comex/llama.cpp/tree/convert-script) which supports the conversion of HuggingFace models to `ggml` format.

We recommend the use of Python 3.10.10 for `convert.py` since we encountered bugs with Python 3.9.5.
> TypeError: 'staticmethod' object is not callable

```
# Clone the specific fork 
git clone --branch convert-script https://github.com/comex/llama.cpp.git
cd llama.cpp
make

# Install Python dependencies
python3 -m pip install -r requirements.txt

# Convert the 7b model to ggml fp16 format
python3 convert.py models/alpaca/pytorch_model.bin

# Quantize the model to 4-bits (using method 2 = q4_0)
./quantize models/alpaca/ggml-model-f16.bin models/alpaca/ggml-model-q4_0.bin 2 

# Run instruction mode with Alpaca
./main -m ./models/alpaca/ggml-model-q4_0.bin --color -f ./prompts/alpaca.txt -ins -b 256 --top_p 0.95 --top_k 50 --temp 0.7 --repeat_penalty 1 -t 7
```

Now you can talk to your own Chatbot!

<details>
<summary><b>Alpaca-7b</b> </summary>
<div align="center">
    <img width="80%" alt="alpaca" src="https://user-images.githubusercontent.com/31032829/230761319-7efa4029-2512-4a8d-b39e-e8a9c263abe5.png">
    <img width="80%" alt="alpaca" src="https://user-images.githubusercontent.com/31032829/230761136-9b126539-af63-48b5-9333-4508e80a5fc3.png">
    <p class="image-caption">Caption: Alpaca cannot respond to the hints.</p>
</div>
</details>
    

<details>
<summary><b>ParroT-Hint-7b</b> </summary>
<div align="center">
    <img width="80%" alt="alpaca" src="https://user-images.githubusercontent.com/31032829/230761459-e0af3d2c-8ce5-446b-8387-a6e66c5c1d62.png">
    <img width="80%" alt="alpaca" src="https://user-images.githubusercontent.com/31032829/230760669-8a05e052-76e6-4123-9992-869cd8c83d83.png">
    <p class="image-caption">Caption: ParroT responds to the hints as expected.</p>
</div>
</details>


## Public Impact

[![Star History Chart](https://api.star-history.com/svg?repos=wxjiao/ParroT&type=Date)](https://star-history.com/#wxjiao/ParroT&Date)


### Acknowledgement
This project cannot be developed without the following resources:
- Meta AI `LLaMA`: https://github.com/facebookresearch/llama
- BigScience `Bloomz`: https://huggingface.co/bigscience/bloom
- HuggingFace developers on `LLaMA`: https://github.com/huggingface/transformers/pull/21955
- Stanford `Alpaca`: https://github.com/tatsu-lab/stanford_alpaca
- OptimalScale: https://github.com/OptimalScale/LMFlow
- `llama.cpp` by [@ggerganov](https://github.com/ggerganov/llama.cpp) and [@comex](https://github.com/comex/llama.cpp)


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
