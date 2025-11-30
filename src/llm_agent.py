import openai
import os

class LLMAgent:
    def __init__(self, model, api_key):
        self.model = model
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    def call(self, prompt_template, input_text, categories=None):
        """
        Call the LLM with the given prompt template and input.

        Args:
            prompt_template (str): Prompt template with {input} and {categories} placeholders.
            input_text (str): Input text to replace {input}.
            categories (list): List of categories to replace {categories}.

        Returns:
            str: LLM response.
        """
        prompt = prompt_template.replace("{input}", input_text)
        if categories:
            prompt = prompt.replace("{categories}", str(categories))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()