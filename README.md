# ms-aai-520-final-project

## Running Locally

Fine tuning:

    docker compose run --rm fine-tune

Inference:

    docker compose run --rm -i inference

Serving the API (from the `src` folder):

    beam serve api.py:web_server

# Deploying the App

To deploy the app to beam.cloud (from the `src` folder):

    beam deploy api.py:web_server