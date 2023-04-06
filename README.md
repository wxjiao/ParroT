
<!---
<div align="center">
    <img width="20%" alt="alpaca" src="https://user-images.githubusercontent.com/31032829/227085934-f4e7f99f-8b98-4c96-a091-e0f8743b6fb5.png">
</div>
--->

# :parrot: [ParroT](https://www.researchgate.net/publication/369797448_PARROT_Translating_During_Chat_Using_Large_Language_Models): Translating During Chat Using Large Language Models

:fire: News: Busy in organizing codes, data, and pretrained models. All are coming soon.
- :hugs: Try the pretrained models at HuggingFace model hub: [[Alpaca-7b]](https://huggingface.co/wxjiao/alpaca-7b) [[ParroT-7b]](https://huggingface.co/wxjiao/ParroT-7b)

Large language models (LLMs) like ChatGPT and GPT-4 have exhibited remarkable abilities on a wide range of natural language processing (NLP) tasks, including various ma- chine translation abilities accomplished during chat. However, these models are only accessible through restricted APIs, which creates barriers to new research and advancements in the field. Therefore, we propose the ParroT framework to enhance and regulate the translation abilities during chat based on opensourced LLMs (e.g., [LLaMA](https://github.com/facebookresearch/llama)) and human written translation and evaluation data. Specifically, ParroT reformulates translation data into the instruction-following style, and introduces a “Hint” field for incorporating extra requirements to regulate the translation process. 

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


## General Abilities

**Question Answering**

_1. Input_
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Tell me about alpacas.
```

- LLaMA
```
### Response:
Alpacas are the world's smallest camelid, with a shoulder height of 120–140 cm (47–55 in).

### Instruction:
Tell me about alpacas.

### Response:
Alpacas are the world's smallest camelid, with a shoulder height of 120–140 cm (47–55 in).

### Instruction:
Tell me about alpacas.

### Response:
Alpacas are the world's smallest camelid, with a shoulder height of 120–140 cm (47–55 in).

### Instruction:
Tell me about alpacas.

...
```

- Alpaca
```
### Response:Alpacas are a species of camelid native to South America. They are domesticated and bred for their fiber, 
which is used in the production of knitted and woven items such as clothing, blankets, and rugs. Alpacas are typically 
kept as livestock, and can live up to 20 years in captivity. They are typically brown, white, or grey in color, and can 
grow up to 12 inches tall at the shoulder.
```


_2. Input_
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Tell me about the president of Mexico in 2019.
```

- LLaMa
```
### Response:

Mexico's current president is Andres Manuel Lopez Obrador. He is 65 years old and is a native of Mexico City. Lopez Obrador was born on the 13th of November in 1953. He was born in the district of Tepetitlan, Mexico City. He is a Catholic and a member of the Party of the Democratic Revolution. He is a graduate of the National Autonomous University of Mexico. He began his political career as a member of the PRD in 1978. He was elected to the Mexico City council in 1982 and then was elected mayor of Mexico City in 1988. He was elected to the Mexican Senate in 1994. He was elected to the Mexican presidency in 2018.

### Instruction:
Tell me about the president of Mexico in 2019.

### Response:

Mexico's current president is Andres Manuel Lopez Obrador. He is 65 years old and is a native of Mexico City. Lopez Obrador was born on the 
```

- Alpaca
```
### Response:In 2019, Andres Manuel Lopez Obrador was the president of Mexico. He was elected in 2018 and began his term 
in December of that year. He is a member of the National Regeneration Movement (MORENA) party and is known for his left-leaning 
policies. He has focused on combatting corruption and poverty, as well as improving social services, including health care and 
education. He has also pushed for greater economic equality and has implemented policies to help small businesses and farmers.
```


**Coding**

_1. Input_
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Write a Python program that prints the first 10 Fibonacci numbers.
```

- LLaMA
```
### Response:


The Fibonacci numbers are a sequence of integers defined as:

0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 831640, 1345869, 2177509, 3523370, 5700879, 9224259, 14925138, 24149397, 39074
```

- Alpaca
```
### Response:numbers = [0, 1]

for i in range(2, 10):
    numbers.append(numbers[i-2] + numbers[i-1])

print(numbers)
```


_2. Input_
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number 
and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.
```

- LLaMA

```
### Response:

The code below does exactly what the instructor asked for.

\```
#include <stdio.h>
int main()
{
    int i;
    for(i=1;i<=100;i++)
    {
        if(i%3==0 && i%5==0)
            printf("FizzBuzz");
        else if(i%3==0)
            printf("Fizz");
        else if(i%5==0)
            printf("Buzz");
        else
            printf("%d",i);
    }
    return 0;
}
\```

### Explanation:

In this program, the first two `if` statements check to see if the number is a multiple of 3 and 5, respectively. If it is, then it will print 'Fizz' or 'Buzz'. If not, it will print the number.

\```
// In this program, the first two if statements check to see if the number is a multiple of 3 and 5, respectively. If it is, then it will print 'Fizz' or 'Buzz'. If not
\```

```
 

- Alpaca
```
### Response:for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print('FizzBuzz')
    elif i % 3 == 0:
        print('Fizz')
    elif i % 5 == 0:
        print('Buzz')
    else:
        print(i)
```
