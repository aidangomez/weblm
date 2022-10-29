import os
import cohere
import openai
import json
import numpy as np
import textwrap
import re
from time import time,sleep


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

co= cohere.Client(open_file('/content/WebLM_interactive_src/WebLM_interactive/cohereapikey.txt'))

def personal_assistant(prompt):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = co.generate(
                model='xlarge',
                prompt= prompt,
                max_tokens=200,
                temperature=1.8,
                k=0,
                p=0.65,
                frequency_penalty=0.15,
                presence_penalty=0.15,
                stop_sequences=[],
                return_likelihoods='NONE')
            text_response = response.generations[0].text.strip()
            text_response = re.sub('\s+', ' ', text_response)
            filename = '%s_log.txt' % time()
            with open('response_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text_response)
            with open('response.txt', 'w') as f:
                f.write(text_response)
            return text_response
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "error: %s" % oops
            print('Error communicating with Cohere:', oops)
            sleep(1)


openai.api_key = open_file('/content/WebLM_interactive_src/WebLM_interactive/openaiapikey.txt')

def code_generation(prompt):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
              model="code-davinci-002",
              prompt="/* create python code using the selenuim library for the following list of tasks." + prompt +"*/",
              temperature=0.8,
              max_tokens=2500,
              top_p=1,
              frequency_penalty=0.5,
              presence_penalty=0.5
            )
            text = response['choices'][0]['text'].strip()
            text= re.sub('\%s+',' ', text)
            filename = '%s_log.txt' % time()
            with open('code_gen_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            with open('generated_code.txt', 'w') as f:
                f.write(text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


if __name__ == '__main__':
    while True:
        query = input("Enter your question here: ")
        query= query.encode(encoding='ASCII',errors='ignore').decode()
        prompt = open_file('prompt.txt').replace('<<QUERY>>', query)
        print(prompt)
        answer = personal_assistant(prompt)
        response= open_file('response.txt').replace('-', '\n')
        # response becomes the new prompt, saves as 'response.txt'
        with open('response.txt', 'w') as f:
          f.write(response)
        
        # sometimes the response tails off about random stuff. just delete the last line
        os.system('sed -i "$ d" {0}'.format('response.txt'))
        print('\n\n', answer)
        code_gen = code_generation(response)
        print('\n\n''This is the generated code:''\n',code_gen)
            