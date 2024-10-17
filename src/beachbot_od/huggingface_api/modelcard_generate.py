import json
from huggingface_hub import ModelCard, ModelCardData
import os
import pandas as pd
import numpy as np
import argparse


def load_config(config_filename):
    # Get the absolute path of the config file relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    config_path = os.path.join(parent_dir, config_filename)

    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def generate_results_table(evaluation_csv="evaluation.csv", markdown_file="README.md"):

    # Convert the CSV string to a pandas DataFrame
    data = pd.read_csv(evaluation_csv)

    # Replace 'nan' strings with actual np.nan for better representation
    data.replace("nan", np.nan, inplace=True)

    # Generate Markdown table
    markdown_table = data.to_markdown(index=False)

    # Output the Markdown table
    print(markdown_table)

    # Optionally, write to a Markdown file
    with open(markdown_file, "w") as f:
        f.write(markdown_table)


def create_model_card(config_path, **kwargs):
    # Load defaults from config file
    defaults = load_config(config_path)

    # Handle 'card_data' separately as it's an object
    defaults["card_data"] = ModelCardData(**defaults["card_data"])

    # Update with any additional keyword arguments provided
    defaults.update(kwargs)

    # Pass the updated values to from_template
    card = ModelCard.from_template(**defaults)

    # Save the card and print
    card.save("README.md")
    print(card)


if __name__ == "__main__":
    create_model_card()
