import json
import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configuration.json")

class ConfigurationSchema(BaseModel):
    thoughts: str = Field(description="Your step-by-step reasoning on what needs to be changed in the configuration based on the user prompt.")
    user_feedback_guidelines: List[str] = Field(description="List of text guidelines/feedback provided by the user.")
    preferred_sectors: List[str] = Field(description="List of preferred market sectors to focus on.")
    wanted_action: str = Field(default="", description="Explicit action requested by the user for the next trading cycle, e.g., 'sell all google stocks' or 'buy google'. Leave empty if no specific immediate action is requested.")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)

config_llm = llm.with_structured_output(ConfigurationSchema)

from typing import List, Tuple

def process_user_prompt(user_prompt: str) -> Tuple[bool, str]:
    """
    Takes a natural language user prompt, loads the current configuration,
    asks the LLM to update the configuration based on the prompt, and saves it.
    """
    try:
        if not os.path.exists(CONFIG_PATH):
            logger.error(f"Configuration file not found at {CONFIG_PATH}")
            return False, ""
            
        with open(CONFIG_PATH, "r") as f:
            current_config = json.load(f)
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Configuration Translation Agent for an automated trading bot.
            Your job is to read the user's natural language request and update the bot's JSON configuration accordingly.
            
            Here is the current configuration:
            {current_config}
            
            Rules:
            1. If the user gives general feedback or rules (e.g., "I don't like when you buy volatile tech stocks", "always hold apple"), update `user_feedback_guidelines`. You MUST review existing guidelines and resolve any conflicts.
            2. If the user expresses a lack of interest in a sector they currently prefer (e.g., "I don't think I want to invest in food"), simply remove it from `preferred_sectors` if it exists. DO NOT add a guideline against it unless they state clearly they are against it.
            3. If the user suggests focusing on new areas, add them to `preferred_sectors`.
            4. If the user requests an immediate, explicit action for the next cycle (e.g., "sell all google stocks now", "buy google"), set the `wanted_action` field.
            5. Keep the existing configuration values if the user does not mention them.
            6. Provide your detailed reasoning in the `thoughts` field.
            
            Return the complete updated configuration as a strictly valid JSON matching the schema.
            """),
            ("human", "{user_prompt}")
        ])
        
        chain = prompt | config_llm
        updated_config_obj = chain.invoke({
            "current_config": json.dumps(current_config, indent=2),
            "user_prompt": user_prompt
        })
        
        # Save back to file
        config_dict = updated_config_obj.model_dump()
        if "thoughts" in config_dict:
            del config_dict["thoughts"]
            
        with open(CONFIG_PATH, "w") as f:
            json.dump(config_dict, f, indent=4)
            
        logger.info("Successfully updated configuration.json based on user prompt.")
        return True, updated_config_obj.thoughts
    except Exception as e:
        logger.error(f"Failed to process user prompt for configuration: {e}")
        return False, str(e)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_user_prompt(" ".join(sys.argv[1:]))
