# WebLM

Drive a browser with Cohere + some random additions 

Derived from Nat's repo here: https://github.com/nat/natbot
Contributing to the WebLM repo here: https://github.com/nat/aidangomez/weblm with a few random additions

Features:
- uses likelihoods instead of raw generations to guide the model's decision-making
- ability to build a reference set of examples that the LM can make reference to
- ability to fix the LM's mistakes online and add them to the reference set for future use

<<<<<<< HEAD

¯\\_(ツ)_/¯ 
=======
## Set up and running instructions

1. Clone this repository and `cd` into its main working directory.
2. Install [poetry](https://python-poetry.org/docs/#installation)
3. Install dependencies: `poetry install --no-root`
4. Set up playwright: `poetry run playwright install`
5. Run main: `poetry run python -m weblm.main`


## Files to add
1. `specials.json` - You should store sensitive information like "Password": "password" to avoid saving it to `examples.json`. 

>>>>>>> 21dc9f16b700ccca313a4a90f0ddbe780c607029
