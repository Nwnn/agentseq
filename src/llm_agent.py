import openai
import os
import time

class LLMAgent:
    def __init__(self, model, api_key):
        self.model = model
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    def call(self, prompt_template, input_text, categories=None, max_retries=3):
        """
        Call the LLM with the given prompt template and input, with retry on failure.

        Args:
            prompt_template (str): Prompt template with {input} and {categories} placeholders.
            input_text (str): Input text to replace {input}.
            categories (list): List of categories to replace {categories}.
            max_retries (int): Maximum number of retry attempts.

        Returns:
            str: LLM response or error message.
        """
        prompt = prompt_template.replace("{input}", input_text)
        if categories:
            prompt = prompt.replace("{categories}", str(categories))

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                if response and response.choices:
                    return response.choices[0].message.content.strip()
                else:
                    error_msg = "Error: No response from LLM"
            except openai.OpenAIError as e:
                error_msg = f"Error: {str(e)}"
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return error_msg

        return "Error: Max retries exceeded"