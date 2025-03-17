from transformers import pipeline


ner = pipeline('question-answering')
print(
    ner(
        question='What is my job?',
        context='My name is Adil and I work remotely at Anvio as a Software Engineer'
    )
)
