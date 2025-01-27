import os
import json
import pandas as pd
import random
import csv

def load_copa():
    train_data,test_data = [],[]
    with open(f"./data/xcopa/train.csv", "r", encoding="utf-8") as g:
        lines = g.readlines()
    for line in lines[1:1501]:
        line = line.strip().split(',')
        label,premise,question,choice1,choice2 = line[0].strip(),line[2].strip(),line[3].strip(),line[4].strip(),line[5].strip()
        prompt = f'Question:\n{premise} Based on the previous passage, choose the most reasonable {question}.\nA:{choice1}\nB:{choice2}\n\nAnswer:\n'
        if int(label) == 0:
            cor_answer,wro_answer = 'A','B'
        elif int(label) == 1:
            cor_answer,wro_answer = 'B','A'
        train_data.append([prompt,cor_answer,wro_answer])


    with open(f"./data/xcopa/test.en.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        premise,question,choice1,choice2,label = line['premise'],line['question'],line['choice1'],line['choice2'],line['label']
        prompt_question = f'Question:\n{premise} Based on the previous passage, choose the most reasonable {question}.\nA:{choice1}\nB:{choice2}\n\nAnswer:\n'
        if int(label) == 0:
            cor_answer,wro_answer = 'A','B'
        elif int(label) == 1:
            cor_answer,wro_answer = 'B','A'
        test_data.append([prompt_question,cor_answer,wro_answer])
    print(train_data[0],test_data[0])
    return train_data,test_data


def load_storycloze():
    train_data,test_data = [],[]
    with open(f"./data/xstorycloze/spring2016.val.en.tsv.split_20_80_train.tsv", "r", encoding="utf-8") as g:
        lines = g.readlines()
    lines = lines[1:]
    for ind in range(len(lines)):
        line = lines[ind].strip()
        line = line.split('\t')
        sent1, sent2,sent3,sent4,quiz1,quiz2,label = line[1],line[2],line[3],line[4],line[5],line[6],line[7]
        sents = sent1 + ' ' + sent2  + ' ' + sent3 + ' ' + sent4
        prompt = f'{sents}\nQuestion: What is a possible continuation for the story given the following options?\nA: {quiz1} B: {quiz2}\nAnswer:'
        if int(label) == 1:
            cor_answer,wro_answer = 'A','B'
        elif int(label) == 2:
            cor_answer,wro_answer = 'B','A'
        train_data.append([prompt,cor_answer,wro_answer])

        
    with open(f"./data/xstorycloze/spring2016.val.en.tsv.split_20_80_eval.tsv", "r", encoding="utf-8") as f:
         lines = f.readlines()
    lines = lines[1:]
    for ind in range(len(lines)):
        line = lines[ind].strip()
        line = line.split('\t')
        sent1, sent2,sent3,sent4,quiz1,quiz2,label = line[1],line[2],line[3],line[4],line[5],line[6],line[7]
        sents = sent1 + ' ' + sent2  + ' ' + sent3 + ' ' + sent4
        prompt = f'{sents}\nQuestion: What is a possible continuation for the story given the following options?\nA: {quiz1} B: {quiz2}\nAnswer:'
        if int(label) == 1:
            cor_answer,wro_answer = 'A','B'
        elif int(label) == 2:
            cor_answer,wro_answer = 'B','A'
        test_data.append([prompt,cor_answer,wro_answer])
    print(train_data[0],test_data[0])
    return train_data,test_data


def load_SST2():
    train_data,test_data = [],[]
    with open(f"./data/SST/sst2/train.jsonl", "r", encoding="utf-8") as g:
        lines = g.readlines()
    lines = lines[:2000]
    for ind in range(len(lines)):
        line = json.loads(lines[ind])
        text,cor_answer = line['text'].strip(),line['label_text'].strip()
        
        prompt = f"Consider the sentiment expression in this sentence and respond briefly with 'positive' or 'negative'.\n\n{text}\n\nAnswer:"
        if cor_answer == 'positive':
            cor_answer, wro_answer = 'Positive','Negative'
        elif cor_answer == 'negative':
            cor_answer, wro_answer = 'Negative','Positive'
        train_data.append([prompt,cor_answer,wro_answer])

        
    with open(f"./data/SST/sst2/test.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for ind in range(len(lines)):
        line = json.loads(lines[ind])
        text,cor_answer = line['text'].strip(),line['label_text'].strip()
        
        prompt = f"Consider the sentiment expression in this sentence and respond briefly with 'positive' or 'negative'.\n\n{text}\n\nAnswer:"
        if cor_answer == 'positive':
            cor_answer, wro_answer = 'Positive','Negative'
        elif cor_answer == 'negative':
            cor_answer, wro_answer = 'Negative','Positive'
        test_data.append([prompt,cor_answer,wro_answer])
    print(train_data[0],test_data[0])
    return train_data,test_data


def load_SST5():
    train_data,test_data = [],[]
    with open(f"./data/SST/sst5/train.jsonl", "r", encoding="utf-8") as g:
        lines = g.readlines()
    lines = lines[:2000]
    for ind in range(len(lines)):
        line = json.loads(lines[ind])
        text,cor_answer = line['text'].strip(),line['label_text'].strip()
        
        prompt = f"Consider the sentiment expression in this sentence and respond briefly with 'very positive', 'positive', 'neutral', 'negative', and 'very negative'.\n\n{text}\n\nAnswer:"
        if cor_answer == 'positive':
            wro_answer = 'negative'
        elif cor_answer == 'very positive':
            wro_answer = 'very negative'
        elif cor_answer == 'negative':
            wro_answer = 'positive'
        elif cor_answer == 'very negative':
            wro_answer = 'very positive'
        elif cor_answer == 'neutral':
            wro_answer = random.choice(['positive', 'negative'])
        
        train_data.append([prompt,cor_answer,wro_answer])

        
    with open(f"./data/SST/sst5/test.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for ind in range(len(lines)):
        line = json.loads(lines[ind])
        text,cor_answer = line['text'].strip(),line['label_text'].strip()
        
        prompt = f"Consider the sentiment expression in this sentence and respond briefly with 'very positive', 'positive', 'neutral', 'negative', and 'very negative'.\n\n{text}\n\nAnswer:"
        if cor_answer == 'positive':
            wro_answer = 'negative'
        elif cor_answer == 'very positive':
            wro_answer = 'very negative'
        elif cor_answer == 'negative':
            wro_answer = 'positive'
        elif cor_answer == 'very negative':
            wro_answer = 'very positive'
        elif cor_answer == 'neutral':
            wro_answer = random.choice(['positive', 'negative'])
        test_data.append([prompt,cor_answer,wro_answer])
    print(train_data[0],test_data[0])
    return train_data,test_data


def load_boolq():
    train_data,test_data = [],[]
    with open(f"./data/boolq/train.jsonl", "r", encoding="utf-8") as g:
        lines = g.readlines()
    lines = lines[:2000]
    for ind in range(len(lines)):
        line = json.loads(lines[ind])
        question,passage,answer = line['question'].strip(),line['passage'].strip(),str(line['answer']).strip()
        
        prompt = f"Is the answer to the question encapsulated in the passage? Please confirm with 'yes' or 'no'.\n\nPassage: {passage}\n\nQuestion: {question}\n\nAnswer:"
        if answer == 'True':
            cor_answer,wro_answer = 'Yes','No'
        elif answer == 'False':
            cor_answer,wro_answer = 'No','Yes'
        train_data.append([prompt,cor_answer,wro_answer])

        
    with open(f"./data/boolq/dev.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for ind in range(len(lines)):
        line = json.loads(lines[ind])
        question,passage,answer = line['question'].strip(),line['passage'].strip(),str(line['answer']).strip()
        
        prompt = f"Is the answer to the question encapsulated in the passage? Please confirm with 'yes' or 'no'.\n\nPassage: {passage}\n\nQuestion: {question}\n\nAnswer:"
        if answer == 'True':
            cor_answer,wro_answer = 'Yes','No'
        elif answer == 'False':
            cor_answer,wro_answer = 'No','Yes'
        test_data.append([prompt,cor_answer,wro_answer])
    print(train_data[0],test_data[0])
    return train_data,test_data

def load_mmlu(file):
    train_data,test_data = [],[]
    with open(os.path.join('./data/mmlu/test/',file), "r", encoding="utf-8") as g:
        reader = csv.reader(g)

        for id_line, data in enumerate(reader):
            if id_line <= 500:
                question,ans1,ans2,ans3,ans4,cor_answer = data[0],data[1],data[2],data[3],data[4],data[5]
                if cor_answer == 'A':
                    wro_answer = random.choice(['B','C','D'])
                elif cor_answer == 'B':
                    wro_answer = random.choice(['A','C','D'])
                elif cor_answer == 'C':
                    wro_answer = random.choice(['A','B','D'])
                elif cor_answer == 'D':
                    wro_answer = random.choice(['A','B','C'])
                prompt = f"Question: {question}\nWhich of the following answers is correct?\nA. {ans1}\nB. {ans2}\nC. {ans3}\nD. {ans4}\nState the letter corresponding to the correct answer.\nAnswer:"
                train_data.append([prompt,cor_answer,wro_answer])

    test_data = train_data   
    print(train_data[0],test_data[0])
    return train_data,test_data

def load_XNLI():
    train_data,test_data = [],[]
    with open("./data/mnli/xnli.dev.tsv") as f:
        lines = f.readlines()
    copora = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for line in lines[1:]:
        line = line.split('\t')
        lang,label,premise,hypothesis = line[0],line[1],line[6],line[7]
        if lang == 'en':
            prompt = f"Question:\n{premise} Based on the previous passage, is it true that \"{hypothesis}\"? Please confirm with 'Yes', 'No', or 'Maybe'.\n\nAnswer:\n"
            prompt = f'Answer whether the hypothesis is more likely to be true, false, or unclusive based on the given premise.\nPremise: {premise}\nHypothesis: {hypothesis}\nAnswer:'
            if label == 'entailment':
                cor_answer, wro_answer = 'True', random.choice(['False','unclusive'])
            elif label == 'contradiction':
                cor_answer, wro_answer = 'False', random.choice(['True','unclusive'])
            elif label == 'neutral':
                cor_answer, wro_answer = 'unclusive', random.choice(['False','True'])
            train_data.append([prompt,cor_answer,wro_answer])
        
    with open("./data/mnli/xnli.test.tsv") as f:
        lines = f.readlines()
    copora = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for line in lines[1:]:
        line = line.split('\t')
        lang,label,premise,hypothesis = line[0],line[1],line[6],line[7]
        if lang == 'en':
            prompt = f'Answer whether the hypothesis is more likely to be true, false, or unclusive based on the given premise.\nPremise: {premise}\nHypothesis: {hypothesis}\nAnswer:'
            if label == 'entailment':
                cor_answer, wro_answer = 'True', random.choice(['False','unclusive'])
            elif label == 'contradiction':
                cor_answer, wro_answer = 'False', random.choice(['True','unclusive'])
            elif label == 'neutral':
                cor_answer, wro_answer = 'unclusive', random.choice(['False','True'])
            test_data.append([prompt,cor_answer,wro_answer])
            
    print(train_data[0],test_data[0])
    return train_data,test_data

def load_winogrande():
    train_data,test_data = [],[]
    with open(f"./data/winogrande/train_m.jsonl", "r", encoding="utf-8") as g:
        lines = g.readlines()
    lines = lines[:2000]
    for ind in range(len(lines)):
        line = json.loads(lines[ind])
        question,option1,option2,answer = line['sentence'].strip(),line['option1'].strip(),line['option2'].strip(),line['answer'].strip()
        prompt = f"Please fill in the blanks. Write A or B as the answer.\n\nSentence: {question}\nA. {option1}\nB. {option2}\nAnswer:"
        if answer == '1':
            cor_answer,wro_answer = 'A','B'
        elif answer == '2':
            cor_answer,wro_answer = 'B','A'
        train_data.append([prompt,cor_answer,wro_answer])

        
    with open(f"./data/winogrande/dev.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for ind in range(len(lines)):
        line = json.loads(lines[ind])
        question,option1,option2,answer = line['sentence'].strip(),line['option1'].strip(),line['option2'].strip(),line['answer'].strip()
        prompt = f"Please fill in the blanks. Write A or B as the answer.\n\nSentence: {question}\nA. {option1}\nB. {option2}\nAnswer:"
        if answer == '1':
            cor_answer,wro_answer = 'A','B'
        elif answer == '2':
            cor_answer,wro_answer = 'B','A'
        test_data.append([prompt,cor_answer,wro_answer])
    print(train_data[0],test_data[0])
    return train_data,test_data