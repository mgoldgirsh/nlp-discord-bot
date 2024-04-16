from transformers import (
    Text2TextGenerationPipeline,
    AutoModelForCausalLM, AutoTokenizer
)
from evaluate import evaluator
from pathlib import Path
from dataset import DialogDataset

# Define your model and tokenizer
model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(f"{Path(__name__).absolute.parent.parent}/discord_gpt_model")
dialog_dataset = DialogDataset("dialog_dataset", model_name)


# Initialize the Text2TextGenerationPipeline
generator = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)

# Generate summaries for the input text
generated_summaries = generator(dialog_dataset.validation_tokenized_data,
                                max_length=100, num_return_sequences=1
                                )

# Initialize the Text2TextGenerationEvaluator
evaluator = evaluator("text-generation")

# Evaluate the generated summaries against the reference summaries
evaluation_results = evaluator.evaluate(generated_summaries)

# Print evaluation results
print(evaluation_results)