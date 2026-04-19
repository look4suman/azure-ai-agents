import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import kernel_function

load_dotenv()

kernel = Kernel()

kernel.add_service(
    AzureChatCompletion(
        service_id="default",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_COMPLETION_MODEL"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
)

# -----------------------------
# Functions
# -----------------------------


class WriterPlugin:

    @kernel_function(name="summarise")
    async def summarise(self, input: str) -> str:
        return await kernel.invoke_prompt(f"Summarise this:\n{input}")

    @kernel_function(name="format_email")
    def format_email(self, email_id: str, text: str) -> str:
        return f"""
        To: {email_id}

        Body:
        {text}
        """


plugin = kernel.add_plugin(WriterPlugin(), plugin_name="writer")

# -----------------------------
# Flow (EXPLICIT)
# -----------------------------


async def run():
    with open("../data/chatgpt.txt", "r") as f:
        text = f.read()

    # Step 1: summarise
    summary = await kernel.invoke(
        plugin["summarise"],
        input=text
    )

    # Step 2: format email
    email = await kernel.invoke(
        plugin["format_email"],
        email_id="look4suman@gmail.com",
        text=str(summary)
    )
    print(email)


asyncio.run(run())
