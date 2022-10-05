import re
import os
from time import time,sleep
from uuid import uuid4
import smtplib
from email.message import EmailMessage
from datetime import datetime

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def listToString(s):
    str1 = " "   
    return (str1.join(s))  

openai.api_key = open_file('openaiapikey.txt')


def gpt3_completion(prompt, engine='text-davinci-002', temp=1.0, top_p=1.0, tokens=2000, freq_pen=0.0, pres_pen=0.0, stop=['asdfasdf', 'asdasdf']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def email_alert(subject, body, to):
  msg= EmailMessage()
  user = "alerts.hackathon@gmail.com"
  #account password: -
  # Generated app password
  password = "-"
  msg.set_content(body)
  msg['from']=user
  msg['to'] = to
  msg['subject'] = subject
  
  server = smtplib.SMTP('smtp.gmail.com', 587)
  server.starttls()
  server.login (user, password)
  server.send_message(msg)
  server.quit()

# email_alert("Fire! Fire! Fire!", "To whom it may concern, it appears there seems to be a fire on the premises.", "-@gmail.com")
# classification_label= "wheelchair"

if __name__ == '__main__':
    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    scenario = open_file('input.txt')
    prompt = open_file('prompt_to_email.txt').replace('<<_TOPIC_>>', scenario)
    print('\n\n==========\n\n', prompt)
    completion = gpt3_completion(prompt)
    filename = scenario.replace(' ','').lower()[0:10] + str(time()) + '.txt'
    output = scenario.strip() + '\n\nResponse:\n\n' + completion
    print('\n\n', completion)
    save_file('sample_emails/%s' % filename, output)
    subject = []
    email= []
    recipient= []
    # phone_number= []
    dir = 'sample_emails/'
    response = dir + filename
    with open(response, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip() # remove "\n" from line
            if line.startswith("Subject:"):
                subject.append(line[8:])
            if line.startswith("Email:"):
                email.append(line[6:])
            if line.startswith("Recipient:"):
                recipient.append(line[10:])
            # if line.startswith("Phone:"):
            #     phone_number.append(line[6:])
    subject= listToString(subject)
    email= listToString(email)
    recipient= listToString(recipient)
    email_alert(subject, email, recipient)
    # email_alert(subject, email, phone_number)