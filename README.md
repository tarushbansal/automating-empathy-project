# Automating Empathy in Dialogue Systems
This is the official respository for the *Automating Empathy in Dialogue Systems* project. A complete technical report is available [here](). All our model checkpoints can be found [here](https://drive.google.com/drive/folders/1RQAFP4HGK_JXgzJMtWjrQDbTW_CMkDrq?usp=share_link). In case of any queries, please contact tb662@cam.ac.uk.

Please run ```pip install -r requirements.txt``` to install all third-party libraries required by the framework.

To setup the EPITOME, EMO-ACC, and INTENT-ACC models, use the following steps:
1. Download all files from the [Google Drive](https://drive.google.com/drive/folders/1PXqmv-MZ1uphHvV81htuAhid2uKGGeGd) hosted by Lee et al., 2022. 
2. Create a directory called ```empathy_metric_models``` in your root folder. 
3. Create a sub-directory in ```empathy_metric_models``` called ```epitome_models``` and place the ```fine_tuned_ER.pth```, ```fine_tuned_EX.pth```, and ```fine_tuned_IP.pth``` files in this sub-directory.
4. Create a sub-directory in ```empathy_metric_models``` called ```emo_classifier```. Unpack ```emotion.tar.gz``` and place all its contents in this sub-directory.
5. Create a sub-directory in ```empathy_metric_models``` called ```intent_classifier```. Unpack ```empintent.tar.gz``` and place all its contents in this sub-directory.

We provide some example commands for the training and testing framework below. Additional command line arguments can also be included. Simply refer to the available list of arguments in the ```parse_args``` function of the corresponding python script. We also have additional scripts such as ```test_gpt3.py```, ```interact.py```, ```evaluate.py```, ```compare.py```, and ```multi-turn.py``` to aide with testing and evaluation as well as interact with models. Please refer to these scripts to learn how to use them.

# Fine-tuning a dialogue model
Following is an example command:

```python3 src/train_dialogue_model.py --dataset_dir datasets/empathetic_dialogues --model GODEL_LARGE --output_dir ~/fine_tuned_models/GODEL_LARGE --max_epochs 5 --batch_size 8 --initial_lr 0.00001```

To add a custom model, follow these steps:
1. Add the model class inherited from ```EncoderDecoderModel``` or ```DecoderModel``` to ```src/dialogue_models.py``` with appropriately defined functions (see ```base_classes.py``` for all top-level functions for the model class)
2. Add the tokenizer class inherited from ```TokenizerBase``` or ```HuggingFaceTokenizerBase``` to ```src/custom_tokenizers.py``` with appropriately defined functions (see ```base_classes.py``` for all top-level functions for the tokenizer class)
3. Add a model configuration in ```configs.json``` with any custom name and the corresponding model and tokenizer class as well as any additional arguments. The format can be copied from some pre-defined configurations.

A custom dataset can be added by splitting the dataset directory into ```train```, ```val```, and ```test``` subdirectories and formatting the data in each directory in the following categories which should be self-explanatory: ```contexts.json```, ```targets.json```, ```instructions.json```(_optional_), and ```knowledge.json``` (_optional_). See ```datasets/empathetic_dialogues``` or ```datasets/blended_skill_talk``` for reference.

# Fine-tuning a reward model
Following is an example command: 

```python3 src/train_reward_model.py --dataset_dir datasets/reward_dataset --model GODEL_LARGE --output_dir ~/reward_models/GODEL_LARGE --max_epochs 1 --initial_lr 0.00001```. 

Batch size support is not included yet. Custom models are added exactly as discussed in the last section. To add a custom dataset, specify a ```train``` subdirectory in the dataset directory formatted in the following categories: ```contexts.json```, ```responses.json```,  and ```ratings.json```. We use the same dataset for validation and have not yet included support for testing reward models. See ```datasets/reward_dataset``` for reference.

# Reinforcement Learning using PPO
Following is an example command: 

```python3 src/train_rlhf_pipeline.py --dataset_dir datasets/empathetic_dialogues --pretrained_model_dir ~/fine_tuned_models/GODEL_LARGE/tensorboard_logs/version_0 --reward_model_dir ~/reward_models/GODEL_LARGE/tensorboard_logs/version_0 --output_dir ~/fine_tuned_models/GODEL_LARGE_RLHF --max_epochs 5 --batch_size 8 --initial_lr 0.00001 --beam_width 1 --sample --top_p 0.9 --top_k 50```. 

# Testing
Following is an example command: 

```python3 src/test.py --dataset_dir datasets/empathetic_dialogues --pretrained_model_dir ~/fine_tuned_models/GODEL_LARGE_RLHF/tensorboard_logs/version_0 --emo_classifier_dir ~/empathy_metric_models/emo_classifier --intent_classifier_dir ~/empathy_metric_models/intent_classifier --epitome_dir ~/empathy_metric_models/epitome_models --reward_model_dir ~/reward_models/GODEL_LARGE --batch_size 128```. 



