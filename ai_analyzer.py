import openai
from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL
import calendar


class AIAnalyzer:
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.month_mapping = {
            month.lower(): index
            for index, month in enumerate(calendar.month_name)
            if month
        }

    def _preprocess_query(self, query):
        query_lower = query.lower()
        for month_name, month_num in self.month_mapping.items():
            if month_name in query_lower:
                query = query.replace(month_name, str(month_num))
                query = query.replace(month_name.capitalize(), str(month_num))
        return query

    def _clean_generated_code(self, code):
        lines = code.strip().split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("```") and not line.startswith("#"):
                line = line.replace("`", "")
                cleaned_lines.append(line)
        return " ".join(cleaned_lines)


def generate_query(self, natural_query, df_info):
    try:
        processed_query = self._preprocess_query(natural_query)
        response = self.client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional financial data analyst and a Python expert. "
                        "Your task is to convert natural language queries into precise Pandas DataFrame operations. "
                        "Guidelines:\n"
                        "- Use the variable name 'df' for the DataFrame.\n"
                        "- The DataFrame contains the following columns: "
                        "date, amount, category, type, year, month, quarter.\n"
                        "- Use numeric values for months (1-12) instead of month names.\n"
                        "- Always return the executable Python code in a single line.\n"
                        "- Do not include any explanations, comments, or markdown in the response.\n"
                        "- The output must be syntactically correct and ready to be passed to eval()."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Convert this natural language query to a Pandas operation: '{processed_query}'",
                },
            ],
        )
        generated_code = response.choices[0].message.content.strip()
        return self._clean_generated_code(generated_code)
    except Exception as e:
        return f"Error generating query: {str(e)}"


def format_result_as_text(self, result, query):
    try:
        response = self.client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial analyst assistant specializing in converting raw numerical data into clear and concise natural language responses. "
                        "Your task is to respond with a summary that directly answers the query, integrating numerical values from the result. "
                        "Guidelines:\n"
                        "- Be concise and precise.\n"
                        "- Always include the relevant numerical values explicitly.\n"
                        "- Format monetary values with a $ symbol and use commas as needed (e.g., $1,000,000).\n"
                        "- For percentages, include a % symbol (e.g., 15%).\n"
                        "- Avoid adding any recommendations or analysis unless the query explicitly asks for it."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Convert the following query result into a natural language response:\n"
                        f"Query: '{query}'\n"
                        f"Result: {result}"
                    ),
                },
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error formatting response: {str(e)}"


def generate_insights(self, data_summary):
    try:
        response = self.client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional financial analyst with expertise in data interpretation and insight generation. "
                        "Your task is to analyze financial data and provide meaningful insights in a structured and clear manner. "
                        "Focus on identifying trends, anomalies, and key performance metrics that can help the user make decisions."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Here is the financial data summary: {data_summary}. "
                        "Please analyze this data and provide 3-5 actionable insights, focusing on revenue trends, expense patterns, "
                        "anomalies, and opportunities for improvement."
                    ),
                },
            ],
            
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {str(e)}"
