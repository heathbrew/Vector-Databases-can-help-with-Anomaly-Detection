from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core import SimpleDirectoryReader, ServiceContext, VectorStoreIndex

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

# model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
# model_url = "https://huggingface.co/TheBloke/Llama-2-7B-chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf"
path = "./models/Llama3/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate.from_template(template)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=None,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    # model_path=None,
    model_path=path,
    temperature=0.1,
    max_new_tokens=4096,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to _call_()
    generate_kwargs={},
    # kwargs to pass to _init_()
    # set to at least 1 to use GPU
    n_gpu_layers=35,
    # model_kwargs={"n_gpu_layers": 30},  #28,29,30 layers works best on my setup.
    # transform inputs into Llama2 format
    callback_manager=callback_manager,
    verbose=True,
)

# Define function to generate text
def generate_text(user_input):
    response_iter = llm.invoke(user_input)
    output_string = response_iter

    return output_string

# Example usage
# user_input = "A rap battle between Stephen Colbert and John Oliver"
# generated_response = generate_text(user_input)
# print(generated_response)