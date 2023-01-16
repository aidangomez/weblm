import os
import cohere
import openai
import json
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import subprocess
import whisper
import torch
import textwrap
import re
from time import time,sleep

AUDIO_ON=0

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
                # model='xlarge'
                model='command-beta',
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



def transcribe():
    torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model("base", device=DEVICE)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )

    audio = whisper.load_audio('audio.wav')
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    if max(probs, key=probs.get) == 'en':
      print('this is in english')
      options = whisper.DecodingOptions(fp16 = False)
      result = whisper.decode(model, mel, options)
      print(result.text)
      
      with open('audio.txt', 'w') as f:
        f.write(result.text)

      result = model.transcribe('audio.wav')
      print(result["text"])
      return result["text"]
    else:
      print(f"This is not English. This language is: {max(probs, key=probs.get)}. Will translate to English")
      options = whisper.DecodingOptions(fp16 = False)
      command_transcribe= f'whisper audio.wav --language {max(probs, key=probs.get)}'.encode() + " --output_dir ./transcribe/".encode()
      subprocess.call(command_transcribe, shell=True)

      command_translate= f'whisper audio.wav --language {max(probs, key=probs.get)}'.encode() + " --task translate".encode() + " --output_dir ./translate/".encode()
      subprocess.call(command_translate, shell=True)
      # transcription = model.transcribe('audio.wav')["text"]
      # translation = model.transcribe('audio.wav', language='en')["text"]
      # print(transcription)
      # print(translation)
      # with open('audio.txt', 'w') as f:
      #   f.write(translation)

      # return translation


def record(duration):
    fs = 44100  # this is the frequency sampling; also: 4999, 64000
    seconds = duration  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    print("Say your request:")
    sd.wait()  # Wait until recording is finished
    print("ok, got it! Give me one sec.")
    write('audio.wav', fs, myrecording)  # Save as wav file

if __name__ == '__main__':
    while True:
        if AUDIO_ON == 1:
          print('Recording, say your question:')
          record(8)
          
          ######  whisper produces better results with default parameters.
          # translation_txt = transcribe()
          # translation_txt= translation_txt.encode(encoding='ASCII',errors='ignore').decode()
          # prompt = open_file('prompt.txt').replace('<<QUERY>>', translation_txt)
          ######

          ##### run whisper with subprocess and saves transcriptions and translations in respective folders
          transcribe()
          translation_txt = open_file('/content/WebLM_interactive_src/WebLM_interactive/translate/audio.wav.txt')
          translation_txt= translation_txt.encode(encoding='ASCII',errors='ignore').decode()
          prompt = open_file('prompt.txt').replace('<<QUERY>>', translation_txt)
          #####
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
          

        else:
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
            