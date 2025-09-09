<h1 align="center">
    Robert Phi 3 Mini
</h1>

## Introduction

We all know someone whose chat messages are unmistakably theirs.
I simply ask the question: is 458 Discord chatlogs enough to fine-tune a lightweight SLM to mimic them?


#### Table of Contents
- [Softwares used](#softwares)
- [Methodology](#methodology)
- [process_messages.py](#process-messages)
- [tokenize-dataset.py](#tokenize-dataset)
- [train.py](#train)
- [inference.py](#inference)
- [Results](#results)
- [Future Considerations](#future)

<br></br>


## Softwares used <a name="softwares"></a>

- Python
- Transformers
- PEFT
- DiscordChatExporter
- Ubuntu 24.04

<br></br>


## Methodology <a name="methodology"></a>
The SLM used for this project is Microsoft's Phi 3 Mini 4K Instruct. A "mere" 3.8B parameters

We will be performing 4-bit QLoRA finetuning

The file pipeline goes as follows: process_messages.py -> tokenize_dataset.py -> train.py -> inference.py

<br></br>


## process_messages.py <a name="process-messages"></a>
We must gather the dataset using DiscordChatExporter

We will make two passes; first gathering messages from direct messages, then any replies in a server

This method isn't very robust as the majority of the individual's chatlogs will not be recorded since most of their messages are probably non-replies

While we lack a sophisticated method to gather more data, we can generally be assured that the data we do have is valid due to the nature of conversation

<br></br>


### tokenize_dataset.py <a name="tokenize-dataset"></a>
We now tokenize the data for the computer to read

We do a 90/10 training/validation split to prevent overfitting and biases

<br></br>


### train.py <a name="train"></a>
We use QLoRA 4-bit quantization to efficiently fine-tune the base model

We trained for 3 epochs at 2e-4 learning rate using fp16 precision with rank=16 and alpha=32 for effective 2x training impact with 5% dropout

<br></br>


### inference.py <a name="inference"></a>
We wrap the base model with our LoRA adapter and prompt the model

We run inferencing with temperature=0.7 and top_p=0.9 so our results are somewhat unpredictable and but not completely random or deterministic

<br></br>


### Results <a name="results"></a>
You'll have to trust me on this, but the results turn out to be extremely similar to how the target communicates over text

Think blunt, direct, and extremely straightforward

<img width="1464" height="190" alt="image" src="https://github.com/user-attachments/assets/7a3baf59-8615-4cee-8b2f-3d460d328acf" />

<br></br>


## Future Considerations <a name="future"></a>
In terms of future considerations, it would be interesting to test different temperature and top_p values, specifically to make the output either more or purely deterministic/greedy

It would also be beneficial to gather more data to create a more accurate representation to mimic our target

Perhaps it'd be fun to host this program and integrate a Discord bot with it
