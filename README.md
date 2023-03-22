# Alpaca-MT
Investigating the capability of [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)-style [LLaMA](https://github.com/facebookresearch/llama) models for machine translation.


## Machine Translation

- **Data**: Flores subsets from [Is-ChatGPT-A-Good-Translator](https://github.com/wxjiao/Is-ChatGPT-A-Good-Translator)
- **Systems**: Google Translate, DeepL, ChatGPT, GPT-4, LLaMA, and Alpaca (reproduced with HuggingFace)
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
