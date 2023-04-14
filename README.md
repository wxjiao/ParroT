
<!---
<div align="center">
    <img width="20%" alt="alpaca" src="https://user-images.githubusercontent.com/31032829/227085934-f4e7f99f-8b98-4c96-a091-e0f8743b6fb5.png">
</div>
--->

# :parrot: ParroT: Translating During Chat Using Large Language Models

:fire: **Update**
- Introducing ParroT-LoRA which supports saving and restarting from the checkpoints (base model and lora weights) in the middle of finetuning.
- Setting the default Transformers to `>= 4.28.0.dev0` directly as it merged the PR of LLaMA. With this version on Torch 1.13.1 + CUDA 11.7, we find the finetuning process could be a bit faster (~18%) than our `v1.0.0` implementation. 

:star: **Highlight** :star:
- :hugs: Try the pretrained models at HuggingFace model hub: [[Alpaca-7b]](https://huggingface.co/wxjiao/alpaca-7b), [[ParroT-7b]](https://huggingface.co/wxjiao/ParroT-7b), [[ParroT-Hint-7b]](https://huggingface.co/wxjiao/ParroT-Hint-7b)
- :page_facing_up: The preprint is available now on arxiv, refer to it for more details: [[paper]](https://arxiv.org/abs/2304.02426) 


## ParroT

> Parrots are smart birds that can respond to simple commands or questions. The question is whether they're just mimicking, or really intelligent enough to communicate with humans. This is similar to what we currently speculate about LLMs.

> Promoting the good is essential, but punishing the evil is also necessary to ensure that goodness prevails. Similarly, aligning LLMs with human feedbacks is exactly to learn from correct examples and discriminate erroneous examples.

Large language models (LLMs) like ChatGPT and GPT-4 have exhibited remarkable abilities on a wide range of natural language processing (NLP) tasks, including various machine translation abilities accomplished during chat. However, these models are only accessible through restricted APIs, which creates barriers to new research and advancements in the field. Therefore, we propose the **ParroT** framework to enhance and regulate the translation abilities during chat based on open-sourced LLMs (e.g., [LLaMA](https://github.com/facebookresearch/llama)) and human written translation and evaluation data. Specifically, ParroT reformulates translation data into the instruction-following style, and introduces a “Hint” field for incorporating extra requirements to regulate the translation process.

<div align="center">
    <img width="60%" alt="LLMs-MT" src="https://user-images.githubusercontent.com/31032829/230255125-bcf7393c-fd3c-4210-a3c6-60dc86a9721d.png">
    <p class="image-caption">Figure 1: Framework of ParroT. Hints are (optional) extra requirements to regulate the translation process.</p>
</div>


## Configurations

### Datasets

- Train Data: data_alpaca_hf.json, [data_parrot_hf.json](https://drive.google.com/file/d/1pQmj-eFwHycSkQtuAB3OKF47bHPxDVon/view?usp=share_link)
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

We develop ParroT based on LLaMA with HuggingFace's transformers library.

Framework Versions:
- Python 3.8.12
- Pytorch 1.13.1+cu117
- Transformers 4.28.0.dev0 
- Peft 0.2.0
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
We did not change any arguments so it would be easy to get started if you are familiar with `run_clm.py`. Also, this script supports data streaming, which might be helpful for handling larger datasets.
[DeepSpeed ZeRO stage 3](https://github.com/microsoft/DeepSpeed) is adopted for model parallel.

LLaMA-7b:
- Original weights for the LLaMA models can be obtained by filling out this [Form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)
- Convert the LLaMA weights into the HuggingFace format by following the instructions in this [Doc](https://huggingface.co/docs/transformers/main/model_doc/llama)

Example usage on 8 V100 by 1 node:
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
    --deepspeed deepspeed_config.json \
    --model_name_or_path ${model_path} \
    --train_file data/data_parrot_hf.json \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
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

<!---
Note: You may try `--gradient_checkpointing True` to reduce memory burden and increase `--per_device_train_batch_size` to speedup the finetuning process.
--->

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


### Finetuned LLMs and Results

Currently, we finetuned the following LLMs for ParroT with the evaluation mainly on WMT22 test sets.

- [x] LLaMA-7b
- [x] Bloomz-mt-7b
- [x] ParroT-LoRA
- [ ] 8bit Training


<div align="center">
    <img width="70%" alt="alpaca" src="https://user-images.githubusercontent.com/31032829/231058559-9cd66740-1d86-4d77-b8d6-c7b4a5837684.png">
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

**Alpaca-7b**

<div align="center">
    <img width="80%" alt="alpaca" src="https://user-images.githubusercontent.com/31032829/230761319-7efa4029-2512-4a8d-b39e-e8a9c263abe5.png">
    <img width="80%" alt="alpaca" src="https://user-images.githubusercontent.com/31032829/230761136-9b126539-af63-48b5-9333-4508e80a5fc3.png">
    <p class="image-caption">Caption: Alpaca cannot respond to the hints.</p>
</div>

**ParroT-Hint-7b**

<div align="center">
    <img width="80%" alt="alpaca" src="https://user-images.githubusercontent.com/31032829/230761459-e0af3d2c-8ce5-446b-8387-a6e66c5c1d62.png">
    <img width="80%" alt="alpaca" src="https://user-images.githubusercontent.com/31032829/230760669-8a05e052-76e6-4123-9992-869cd8c83d83.png">
    <p class="image-caption">Caption: ParroT responds to the hints as expected.</p>
</div>



## Public Impact

### Acknowledgement
This project cannot be developed without the following resources:
- Meta AI `LLaMA`: https://github.com/facebookresearch/llama
- HuggingFace developers on `LLaMA`: https://github.com/huggingface/transformers/pull/21955
- Stanford `Alpaca`: https://github.com/tatsu-lab/stanford_alpaca
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
