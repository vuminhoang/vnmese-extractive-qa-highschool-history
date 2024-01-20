from bm25_semantic import clean_sem, bm25_search_s, three_sub_relevant
from extractive_qa_mrc.infer import tokenize_function, data_collator, extract_answer
from extractive_qa_mrc.model.mrc_model import MRCQuestionAnswering
from transformers import AutoTokenizer

model_checkpoint = "nguyenvulebinh/vi-mrc-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = MRCQuestionAnswering.from_pretrained(model_checkpoint)

def get_answer(question, context):
    answers = []
    if len(context.split(" ")) > 295:
        list_contexts = overlap_context(context, 50, max_size=295)

        for cont in list_contexts:
            cont = cont.split()
            if 'c.1' in cont[0]:
                cont = ' '.join(cont[1:])
            else:
                cont = ' '.join(cont[0:])
            cont = clean_context(cont)
            QA_input = {
                'question': question,
                'context': cont
            }

            inputs = [tokenize_function(QA_input, tokenizer)]
            inputs_ids = data_collator(inputs, tokenizer)
            outputs = model(**inputs_ids)
            answer = extract_answer(inputs, outputs, tokenizer)
            answers.append(answer)
    else:
        context = clean_context(context)
        context = context.split()
        if 'c.1' in context[0]:
            context = ' '.join(context[1:])
        else:
            context = ' '.join(context[0:])
        QA_input = {
            'question': question,
            'context': context
        }
        inputs = [tokenize_function(QA_input, tokenizer)]
        inputs_ids = data_collator(inputs, tokenizer)
        outputs = model(**inputs_ids)
        answers = [extract_answer(inputs, outputs, tokenizer)]

    return answers


def overlap_context(context, overlap_size, max_size=300):
    context_words = context.split(" ")
    len_context = len(context_words)
    sub_len = max_size - overlap_size
    number_sub = (len_context - overlap_size) // sub_len + 1

    sub_contexts = []
    for i in range(number_sub):
        start = i * sub_len
        end = min(start + max_size, len_context)
        sub_context = context_words[start:end]

        sub_context = " ".join(sub_context)

        sub_contexts.append(sub_context)
    return sub_contexts


def clean_context(text):
    text = text.lower()
    punc = '''![]{}'"\<>/?@#$^&*_~'''

    for ele in text:
        if ele in punc:
            text = text.replace(ele, "")
    return text


def clean_answer(answer):
    if ' %' in str(answer):
        answer = str(answer.replace(' %', '%'))
    return answer


def answer_from_model(question, context):
    answer = '-1'
    results = get_answer(question, context)
    for result in results:
        answer = result[0]['answer']
    return answer

def answer_bm25semantic(question):
    question = clean_sem(question)
    arr = bm25_search_s(question,  limit=2)
    context1 = arr[0]
    context2 = arr[1]
    sub1 = three_sub_relevant(question, context1)
    sub2 = three_sub_relevant(question, context2)
    answer = ''
    context = ''
    for i in sub1:
        a =  answer_from_model(question, i)
        if a != '':
            answer = a
            context = i
            return answer, context
    for i in sub2:
        b = answer_from_model(question, i)
        if b!='':
            answer = b
            context = i
        return answer, context

    return answer