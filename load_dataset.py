
def load_data_XNLI():
    langs = ['en','ar', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'vi', 'zh']
    label_dict = {'entailment':'true', 'contradiction':'false', 'neutral':'inconclusive'}
    with open("./data/xnli.test.tsv") as f:
        lines = f.readlines()
    copora = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for line in lines[1:]:
        line = line.split('\t')
        lang,label,sent1,sent2 = line[0],line[1],line[6],line[7]
        if lang in langs:
            ind = langs.index(lang)
            res = (label,sent1,sent2)
            copora[ind].append(res)
        
    length = len(copora[0])
    questions, answers = [],[]
    for i in range(length):
        questions.append([])
        answers.append([])
    for ind in range(length):
        for l in range(len(copora)):
            tup = copora[l]
            label,premise,hypothesis = tup[ind][0],tup[ind][1],tup[ind][2]
            prompt = f'Take the following as truth: {premise}\nThen the following statement: "{hypothesis}" is "true", "false", or "inconclusive"?'
            questions[ind].append(prompt)
            answers[ind].append(label_dict[label])
    question_all, answer_all = [],[]        
    for j in range(len(questions[0])):
        temp_q,temp_a = [],[]
        for i in range(len(questions)):
            temp_q.append(questions[i][j])
            temp_a.append(answers[i][j])
        question_all.append(temp_q)
        answer_all.append(temp_a)
    
    print(len(question_all),len(question_all[0]))
    print(question_all[0])
    print(len(answer_all),len(answer_all[0]))
    print(answer_all[0])      
    return question_all,answer_all


def load_data_xcopa():
    langs = ['en','et','id','it','sw','ta','th','tr','vi','zh']
    questions_en = []
    with open(f".data/xcopa/test.en.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        ques = line['question']
        questions_en.append(ques)
    question_all,answer_all = [],[]
    for lang in langs:
        questions,answers = [],[]
        with open(f"../data/xcopa/test.{lang}.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
        for ind in range(len(lines)):
            line = lines[ind]
            line = json.loads(line)
            premise,choice1,choice2,label = line['premise'],line['choice1'],line['choice2'],line['label']
            question = questions_en[ind]
            prompt_question = f'Here is a premise: "{premise}". A: "{choice1}" B: "{choice2}" What is the {question}? "A" or "B"?'

            if int(label) == 0:
                prompt_answer = 'A'
            elif int(label) == 1:
                prompt_answer = 'B'
            questions.append(prompt_question)
            answers.append(prompt_answer)
        question_all.append(questions)
        answer_all.append(answers)
        
    print(len(question_all),len(question_all[0]))
    print(question_all[0][0])
    print(len(answer_all),len(answer_all[0]))
    print(answer_all[0][0])
    return question_all,answer_all



def load_data_xstorycloze():
    langs = ['en','ar','es','hi','id','ru','sw','zh']
    label_dict = {'1':'A', '2':'B'}
    question_all,answer_all = [],[]
    for lang in langs:
        questions,answers = [],[]
        with open(f"./data/xstorycloze/spring2016.val.{lang}.tsv.split_20_80_eval.tsv", "r", encoding="utf-8") as f:
             lines = f.readlines()
        lines = lines[1:]
        for ind in range(len(lines)):
            line = lines[ind].strip()
            line = line.split('\t')
            sent1, sent2,sent3,sent4,quiz1,quiz2,label = line[1],line[2],line[3],line[4],line[5],line[6],line[7]
            sents = sent1 + ' ' + sent2  + ' ' + sent3 + ' ' + sent4
            sent = f'{sents}\nWhat is a possible continuation for the story given the following options?\nA: {quiz1} B:{quiz2}'
            questions.append(sent)
            answers.append(label_dict[label])
        question_all.append(questions)
        answer_all.append(answers)
            
    print(len(question_all),len(question_all[0]))
    print(question_all[1][0])
    print(len(answer_all),len(answer_all[0]))
    print(answer_all[1][0])
    return question_all,answer_all



def load_data_xwinograd():
    langs = ['en','fr','ja','pt','ru','zh']
    question_all,answer_all = [],[]
    for lang in langs:
        questions,answers = [],[]
        with open(f"./data/xwinograd/test_{lang}.jsonl", "r", encoding="utf-8") as f:
             lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            sents,option1,option2,ans = line['sentence'],line['option1'],line['option2'],line['answer']
            sent = f'{sents}\nReplace the _ in the above sentence with the correct option:\n- {option1}\n- {option2}'                                                                                       
            questions.append(sent)
            if ans == 1:
                answers.append(option1)
            elif ans == 2:
                answers.append(option2)
        question_all.append(questions)
        answer_all.append(answers)
    print(len(question_all),len(question_all[0]))
    print(question_all[-1][0])
    print(len(answer_all),len(answer_all[0]))
    print(answer_all[-1][0])
    return question_all,answer_all



def load_data_flores():    
    langs = ['en', 'ar', 'el', 'es', 'fr', 'hi', 'ru', 'tr', 'vi', 'zh']

    lang_dict = ['eng','ara','ell','spa','fra','hin','rus','tur','vie','zho_simpl']
    lang_name = ['English','Arabic','Greek','Spanish','French','Hindi','Russian','Turkish','Vietnamese','Chinese']
    question_all, answer_all = [],[]
    for lang_id in range(len(lang_dict)):
        questions,answers = [],[]
        lang = lang_dict[lang_id]
        name = lang_name[lang_id]
        with open(f"./data/flores101_dataset/devtest/{lang}.devtest") as f:
            lines = f.readlines()
        for ind in range(len(lines)):
            line = lines[ind].strip()        
            prompt = f'Translate the following sentence from {name} to English: {line}\n'
            questions.append(prompt)
            answers.append(line)
        question_all.append(questions)
        answer_all.append(answers)
    print(len(question_all),len(question_all[0]))
    print(question_all[0])
    print(len(answer_all),len(answer_all[0]))
    print(answer_all[0])      
    return question_all,answer_all



def load_data_wmt23():    
    langs = ['de', 'ja', 'ru', 'uk', 'zh']
    lang_dict = ['deu','jpn','rus','ukr','zho_simpl']
    lang_name = ['German','Japanese','Russian','Ukrainian','Chinese']
    question_all, answer_all = [],[]
    for lang_id in range(len(langs)):
        questions,answers = [],[]
        name = lang_name[lang_id]
        lang = langs[lang_id]
        with open(f"./data/wmt23/generaltest2023.{lang}-en.src.{lang}") as f:
            lines_nonen = f.readlines()
        with open(f"./data/wmt23/generaltest2023.{lang}-en.ref.refA.en") as f:
            lines_en = f.readlines()
        
        for ind in range(len(lines_en)):
            line = lines_nonen[ind].strip()
            line_en = lines_en[ind].strip()
            prompt = f'{name}: {line}\nEnglish:'
            questions.append(prompt)
            answers.append(line_en)
        question_all.append(questions)
        answer_all.append(answers)
    print(len(question_all),len(question_all[0]))
    print(question_all[-1][0])
    print(len(answer_all),len(answer_all[0]))
    print(answer_all[-1][0])      
    return question_all,answer_all



def load_data_xcsqa():    
    langs = ['en','ar', 'de', 'es', 'fr', 'hi','it','ja','nl','pt','ru', 'sw', 'vi', 'zh']
    question_all, answer_all = [],[]
    for lang in langs:
        questions,answers = [],[]
        with open(f"/bask/homes/f/fksv3157/xngs6460-languages/weixuan/data/xcsqa/dev-{lang}.jsonl") as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            question,choices,answer = line['question']['stem'],line['question']['choices'],line['answerKey']
            chos = []
            for cho in choices:
                chos.append(cho['label'] + ': ' + cho['text'])
            choice = ' '.join(chos)
            question = f'Question: {question} {choice}\nAnswer:'
            questions.append(question)
            answers.append(answer)
        question_all.append(questions)
        answer_all.append(answers)
    print(len(question_all),len(question_all[0]))
    print(question_all[-1][0])
    print(len(answer_all),len(answer_all[0]))
    print(answer_all[-1][0])      
    return question_all,answer_all
