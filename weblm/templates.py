

user_prompt_end = ("\n\t(success) the goal is accomplished"
                   "\n\t(cancel) terminate the session"
                   "\nType a choice and then press enter:")
user_prompt_1 = ("Given web state:\n{state}"
                 "\n\nI have to choose between `clicking` and `typing` here."
                 "\n**I think I should{action}**"
                 "\n\t(y) proceed with this action"
                 "\n\t(n) do the other action" + user_prompt_end)
user_prompt_2 = ("Given state:\n{self._construct_state(url, pruned_elements)}"
                 "\n\nSuggested command: {cmd}.\n\t(y) accept and continue"
                 "\n\t(s) save example, accept, and continue"
                 "\n{other_options}"
                 "\n\t(back) choose a different action"
                 "\n\t(enter a new command) type your own command to replace the model's suggestion" + user_prompt_end)
user_prompt_3 = ("Given state:\n{self._construct_state(url, pruned_elements)}"
                 "\n\nSuggested command: {self._cmd}.\n\t(y) accept and continue"
                 "\n\t(s) save example, accept, and continue"
                 "\n\t(back) choose a different action"
                 "\n\t(enter a new command) type your own command to replace the model's suggestion" + user_prompt_end)

state_template = """Objective: $objective
Current URL: $url
Current Browser Content:
------------------
$browser_content
------------------
Previous actions:
$previous_commands"""

help_msg = """Welcome to WebLM!
    
The goal of this project is build a system that takes an objective from the user, and operates a browser to carry it out.

For example:
- book me a table for 2 at bar isabel next wednesday at 7pm
- i need a flight from SF to London on Oct 15th nonstop
- buy me more whitening toothpaste from amazon and send it to my apartment

WebLM learns to carry out tasks *by demonstration*. That means that you'll need to guide it and correct it when it goes astray. Over time, the more people who use it, the more tasks it's used for, WebLM will become better and better and rely less and less on user input.

To control the system:
- You can see what the model sees at each step by looking at the list of elements the model can interact with
- show: You can also see a picture of the browser window by typing `show`
- goto: You can go to a specific webpage by typing `goto www.yourwebpage.com`
- success: When the model has succeeded at the task you set out (or gotten close enough), you can teach the model by typing `success` and it will save it's actions to use in future interations
- cancel: If the model is failing or you made a catastrophic mistake you can type `cancel` to kill the session
- help: Type `help` to show this message

Everytime you use WebLM it will improve. If you want to contribute to the project and help us build join the discord (https://discord.com/invite/co-mmunity) or send an email to weblm@cohere.com"""

prompt_template = """Given:
    (1) an objective that you are trying to achieve
    (2) the URL of your current web page
    (3) a simplified text description of what's visible in the browser window

Your commands are:
    click X - click on element X.
    type X "TEXT" - type the specified text into input X

Here are some examples:

$examples

Present state:
$state
Next Command:"""

prioritization_template = """Here are the most relevant elements on the webpage (links, buttons, selects and inputs) to achieve the objective below:
Objective: buy me toothpaste from amazon
URL: https://www.google.com/search?q=toothpaste+amazon&source=hp&ei=CpBZY5PrNsKIptQP77Se0Ag&iflsig=AJiK0e
Relevant elements:
link 255 role="text" role="text" "toothpaste - Amazon.com https://www.amazon.com › toothpaste › k=toothpaste"
link 192 role="text" role="text" "Best Sellers in Toothpaste - Amazon.ca https://www.amazon.ca › zgbs › beauty"
link 148 role="heading" role="text" "Shop Amazon toothpaste - Amazon.ca Official Site Ad · https://www.amazon.ca/"
---
Here are the most relevant elements on the webpage (links, buttons, selects and inputs) to achieve the objective below:
Objective: book me in for 2 at bar isabel in toronto on friday night
URL: https://www.opentable.ca/r/bar-isabel-toronto
Relevant elements:
select 119 TxpENin57omlyGS8c0YB Time selector restProfileSideBartimePickerDtpPicker "5:00 p.m. 5:30 p.m. 6:00 p.m. 6:30 p.m. 7:00 p.m. 7:30 p.m. 8:00 p.m. 8:30 p.m. 9:00 p.m. 9:30 p.m. 10:00 p.m. 10:30 p.m. 11:00 p.m. 11:30 p.m."
select 114 Party size selector FfVyD58WJTQB9nBaLQRB restProfileSideBarDtpPartySizePicker "1 person 2 people 3 people 4 people 5 people 6 people 7 people 8 people 9 people 10 people 11 people 12 people 13 people 14 people 15 people 16 people 17 people 18 people 19 people 20 people"
button 121 aria-label="Find a time" "Find a time"
---
Here are the most relevant elements on the webpage (links, buttons, selects and inputs) to achieve the objective below:
Objective: email aidan@cohere.com telling him I'm running a few mins late
URL: https://www.google.com/?gws_rd=ssl
Relevant elements:
link 3 "Gmail"
input 10 gLFyf gsfi q text combobox Search Search
---
Here are the most relevant elements on the webpage (links, buttons, selects and inputs) to achieve the objective below:
Objective: buy me a pair of sunglasses from amazon
URL: https://www.amazon.ca/LUENX-Aviator-Sunglasses-Polarized-Gradient/dp/B08P7HMKJW
Relevant elements:
button 153 add-to-cart-button submit.add-to-cart Add to Shopping Cart a-button-input Add to Cart
button 155 buy-now-button submit.buy-now a-button-input
select 152 quantity quantity a-native-dropdown a-declarative "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30"
---
Here are the most relevant elements on the webpage (links, buttons, selects and inputs) to achieve the objective below:
Objective: $objective
URL: $url
Relevant elements:
{element}"""




