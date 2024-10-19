# MS-AAI 520: Dialogue Generation Chatbot
Douglas Code - Group 11

This repo contains the implementation for a chatbot that generates dialogue when prompted with a preceding line.
Source code can be found in the `src` directory, and the LaTeX code for the final paper is in the `final-paper` directory.

Key source files:
* **config.py**: holds configuration settings for the entire project, including file paths and training hyperparameters.
* **api.py**: a beam.cloud-based FastAPI web server for the chatbot
* **inference.py**: the core "chatbot" functionality for generating responses from the trained models.
* **preprocessor.py**: prepares and formats the raw conversation data from the Cornell Movie-Dialog Corpus for model training.
* **fine_tune_model.py**: the process for fine-tuning the base model on the Cornell Movie Dialog dataset.
* **generate_dpo_prompts.py**: uses the OpenAI API to generate dialogue prompts for use in creating DPO training data.
* **generate_dpo_data.py**: uses the OpenAI API to create a dataset of preferred and rejected responses for use in DPO training.
* **dpo.py**: the Direct Preference Optimization (DPO) training process to further tune the model based on the preference data.
* **generate_test_responses.py**: produces test responses using different versions of the model for use in G-Eval evaluation.
* **score_test_responses.py**: uses G-Eval to evaluate the quality of the generated test responses using the OpenAI API.
* **eval_visualizations.ipynb**: visualizations for the outcomes of model evaluation.

## Running Locally

Fine-tuning (or any non-interactive service):

    docker compose run --rm fine-tune

Inference:

    docker compose run --rm -i inference

Serving the API (from the `src` folder):

    beam serve api.py:web_server

# Deploying the App

To deploy the app to beam.cloud (from the `src` folder):

    beam deploy api.py:web_server