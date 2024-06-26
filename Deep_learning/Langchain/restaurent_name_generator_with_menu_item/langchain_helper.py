from secret_key import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

llm = OpenAI(temperature=0.7, api_key=openai)

def generate_restaurant_name_and_items(cuisine):
    # Prompt for generating a restaurant name
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template='I want to open a restaurant for {cuisine} food. Suggest a fancy name for this'
    )
    chain3 = LLMChain(llm=llm, prompt=prompt_template_name, output_key='restaurant_name')

    # Prompt for generating menu items
    prompt_template_menu = PromptTemplate(
        input_variables=['restaurant_name'],
        template='Suggest 20 menu items for {restaurant_name}. Return it as a comma separated value'
    )
    chain4 = LLMChain(llm=llm, prompt=prompt_template_menu, output_key='menu_items')

    seq_chain = SequentialChain(
        chains=[chain3, chain4],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items']
    )

    response = seq_chain.invoke({'cuisine': cuisine})
    return response

