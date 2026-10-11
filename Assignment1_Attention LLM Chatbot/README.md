# COMP5313 Artificial Intelligence — Assignment 1

## Python/PyTorch Chatbot with Adjustable Attention

**Student:** Sumit Rajendra Bhosle  
**Course:** COMP5313 Artificial Intelligence  
**University:** Lakehead University  
**Environment:** Windows 11, Anaconda, Jupyter Notebook, CPU-based PyTorch  
**Main model:** `Qwen/Qwen2.5-1.5B-Instruct`

---

## 1. Project Overview

This project implements a simple chatbot with an adjustable attention mechanism using Python, PyTorch, Hugging Face Transformers and a pretrained Qwen language model.

The main idea is simple:

> **Take a long user message → split it into manageable token chunks → select Local or Global attention → visualize how much token-to-token interaction is allowed → reconstruct the context → generate a response with the LLM.**

The project was built step-by-step in Jupyter Notebook instead of directly writing only the final chatbot. This was intentional. Each important part was tested manually first so that the attention mechanism, tokenization, chunking and response generation could be understood separately before combining everything into the final pipeline.

The assignment asks for three main parts:

1. **Text Chunking Module**
2. **Attention Control Panel**
3. **LLM Engine**

The implementation follows this overall flow:

```text
User Input
    ↓
Tokenization
    ↓
Fixed Token Chunking
    ↓
Attention Control
 ┌───────────────┐
 │               │
Local          Global
 │               │
Sparse          Dense
Mask            Mask
 └───────┬───────┘
         ↓
Attention Coverage / Visualization
         ↓
Context Reconstruction
         ↓
Qwen 2.5 1.5B Instruct
         ↓
Generated Response
```

---

## 2. Assignment Requirement Mapping

| Assignment Requirement | Project Implementation |
|---|---|
| Text Chunking Module | Input is tokenized and divided into fixed 32-token chunks |
| Attention Control Panel | Local / Global selection in notebook and Streamlit GUI |
| Local Attention | Sliding-window mask with window size = 2 |
| Global Attention | Full attention mask |
| Input Split | Long input is divided into uniform token chunks |
| Attention Shift | The selected mode changes the generated attention mask |
| Output Generation | Qwen2.5-1.5B-Instruct generates the chatbot response |
| Visualization | Token representation, attention masks, attention weights and comparison plots |
| GUI | Streamlit application in `app.py` |

The assignment describes the workflow as input splitting, attention shifting and output generation. This project follows that same high-level structure.

---

## 3. Why Qwen2.5-1.5B-Instruct?

The project uses:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

A few model options were considered during development, including smaller models. The final selection was Qwen2.5-1.5B-Instruct because it gives a better balance between response quality and the ability to run locally on a normal student laptop.

The important point is that the project does **not** require an external paid API for inference. The tokenizer and model are downloaded through Hugging Face and then used locally.

### Why not use a much larger model?

A larger LLM would require significantly more memory and/or GPU resources. For this assignment, the goal is to demonstrate the chunking and adjustable-attention concept, not to run a production-scale LLM.

The 1.5B model is therefore a practical middle ground for local experimentation.

---

## 4. Important Technical Clarification

The Local and Global masks in this project are implemented as an **application-level attention demonstration and routing layer**.

The project creates the attention masks, calculates attention coverage, and demonstrates masked attention using PyTorch Q/K/V tensors. However, the custom Local/Global mask is **not injected into Qwen's internal Transformer attention layers** in the final chatbot generation call.

This distinction is important.

The final Qwen generation receives the reconstructed user context in both modes. The selected mask is used to represent and measure the attention strategy at the project level.

This makes the project a clear demonstration of:

- token chunking,
- sparse vs. dense attention structure,
- attention coverage,
- attention weight calculation,
- context preparation, and
- LLM response generation.

A future extension could directly pass a custom attention mask into the model's internal Transformer attention implementation.

---

## 5. Project Structure

The project folder can be kept as:

```text
Assignment1_Attention LLM Chatbot/
│
├── Sumit_Assignment1.ipynb
├── app.py
└── ReadMe.pdf
```

Optional development files can be kept outside the final submission folder.

### Notebook

`Sumit_Assignment1.ipynb`

This is the main development and explanation file. It shows the project being built and tested step-by-step.

### Streamlit application

`app.py`

This is the GUI version of the project. It is independent from the notebook at runtime. Running `app.py` does not execute the notebook.

---

# 6. Environment Used

The notebook records the following environment:

```text
Python       3.12.14
PyTorch      2.14.0+cpu
CUDA         False
Transformers 5.17.0
```

The project was developed for a Windows 11 laptop and tested using CPU execution.

The notebook also uses:

```text
Jupyter Notebook
Matplotlib
Anaconda / Conda
```

The Streamlit GUI additionally requires:

```text
Streamlit
```

---

# 7. Installation

## Step 1 — Install Anaconda

Install Anaconda or Miniconda for Windows.

After installation, open:

```text
Anaconda Prompt
```

## Step 2 — Create the environment

```bash
conda create -n comp5313 python=3.12
```

Activate it:

```bash
conda activate comp5313
```

## Step 3 — Install PyTorch

For the CPU-based setup used in this project:

```bash
pip install torch
```

If a specific PyTorch build is required for the machine, use the installation command recommended by the official PyTorch installation selector.

## Step 4 — Install Transformers

```bash
pip install transformers
```

## Step 5 — Install Matplotlib

```bash
pip install matplotlib
```

## Step 6 — Install Jupyter

```bash
pip install jupyter
```

## Step 7 — Install Streamlit

```bash
pip install streamlit
```

At this point the environment contains the main packages required by the notebook and GUI.

---

# 8. Verify the Environment

Open Anaconda Prompt:

```bash
conda activate comp5313
```

Then run:

```bash
python
```

Inside Python:

```python
import torch
import transformers

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
```

For the original development environment, the notebook reported:

```text
PyTorch: 2.14.0+cpu
CUDA available: False
Transformers: 5.17.0
```

The exact version shown may differ if the environment is recreated later.

---

# 9. Running the Jupyter Notebook

Go to the project directory:

```bash
cd "C:\Stayhard\Lakehead_projects\COMP5313_AI\Assignment1_Attention LLM Chatbot"
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
Sumit_Assignment1.ipynb
```

Run the notebook from the beginning.

## Important

The notebook is intentionally ordered as a learning/development process.

Do not only look at the final chatbot cell. The earlier cells explain how the mechanism was tested.

---

# 10. How the Notebook Was Built

The notebook starts with environment verification.

### Stage 1 — Environment Check

The first cell checks:

- Python version
- PyTorch version
- CUDA availability
- Transformers version

This establishes what hardware/software configuration is actually being used.

### Stage 2 — Model Loading

The project loads:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
```

and then:

```python
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
```

The tokenizer and model are loaded from Hugging Face.

### Stage 3 — First Inference

Before building the attention system, a simple question is sent to the model:

```text
Explain what artificial intelligence is in one sentence.
```

This confirms that the model is working before adding any project-specific processing.

### Stage 4 — Understanding Tokens

The notebook converts the model input into token IDs and displays them visually.

This was done to answer a basic question:

> What does the LLM actually receive?

The model does not directly receive a normal human sentence. It receives token IDs.

---

# 11. Text Chunking Development

The first experiment uses sentence-based chunking.

The notebook checks the number of tokens in each sentence.

The result shows that sentence chunks are **not uniform in size**.

For example, the tested long text produced different sentence token lengths:

```text
31
19
22
21
22
33
18
15
...
```

This leads to the next design decision.

## Fixed Token Chunking

The project then uses:

```text
Chunk size = 32 tokens
```

The tested long text contained:

```text
341 tokens
```

which produced:

```text
11 chunks
```

The first ten chunks contained 32 tokens and the final chunk contained 21 tokens.

This matches the assignment's idea of splitting input into uniform token chunks more closely than sentence-only splitting.

---

# 12. Local Attention

Local attention uses a sliding window.

The project uses:

```text
Window size = 2
```

For a token at position `i`, attention is allowed when:

```text
|i - j| <= 2
```

This creates a diagonal pattern in the attention matrix.

Conceptually:

```text
Nearby tokens → allowed
Far-away tokens → blocked
```

This reduces the number of token-to-token connections compared with a full matrix.

---

# 13. Global Attention

Global attention creates a full matrix of ones.

Conceptually:

```text
Every token → can attend to every other token
```

For a 32-token chunk:

```text
32 × 32 = 1024
```

possible connections are available.

---

# 14. Attention Control Function

The notebook converts the manual experiments into a reusable function:

```python
create_attention_mask(
    sequence_len,
    mode="local",
    window_size=2
)
```

The two available modes are:

```text
local
global
```

If an unsupported mode is supplied, the function raises an error.

This becomes the core of the Attention Control Panel.

---

# 15. Manual Q/K/V Attention Demonstration

Before connecting the attention idea to the chatbot, the notebook demonstrates the actual attention calculation using PyTorch tensors.

The process is:

```text
Query
  +
Key
  ↓
Attention Scores
  ↓
Apply Mask
  ↓
Softmax
  ↓
Attention Weights
  ↓
Weighted Value
  ↓
Attention Output
```

The notebook verifies that blocked positions in Local mode receive:

```text
0.0
```

attention.

It also compares Local and Global attention outputs using the mean absolute difference.

The test produced:

```text
Mean absolute difference:
0.6864795684814453
```

The exact value can change if the random tensors are regenerated.

---

# 16. Long-Context Test

The final development test uses a longer user message about planning a trip around Thunder Bay.

The input contains:

```text
86 tokens
```

and is divided into:

```text
3 chunks
```

with:

```text
Chunk 1 = 32 tokens
Chunk 2 = 32 tokens
Chunk 3 = 22 tokens
```

This is the main example used to demonstrate the multi-chunk workflow.

---

# 17. Attention Coverage

The project calculates the percentage of allowed connections.

For the 3-chunk Thunder Bay test:

### Local

```text
Chunk 1: 154 / 1024 = 15.04%
Chunk 2: 154 / 1024 = 15.04%
Chunk 3: 104 / 484 = 21.49%
```

### Global

```text
Chunk 1: 1024 / 1024 = 100%
Chunk 2: 1024 / 1024 = 100%
Chunk 3: 484 / 484 = 100%
```

This provides a simple numerical way to show the difference between sparse and dense attention.

---

# 18. Context Reconstruction

After processing the chunks, the token IDs are decoded back into readable text.

The chunks are joined together to reconstruct the original context before sending it to the LLM.

This is important because the chatbot should not lose the user's complete message simply because the message was split for processing.

The notebook therefore keeps the original information while using the attention layer to demonstrate the different interaction patterns.

---

# 19. Final Chatbot Pipeline

The reusable chatbot function follows five main stages:

```text
1. Split input into token chunks
        ↓
2. Apply selected attention mode
        ↓
3. Prepare / reconstruct context
        ↓
4. Build Qwen chat prompt
        ↓
5. Generate response
```

The final reusable function is:

```python
generate_chatbot_response(
    user_text,
    mode="local",
    chunk_size=32
)
```

The function returns the generated response and the number of chunks.

---

# 20. Streamlit GUI

The project also includes:

```text
app.py
```

The GUI provides:

- message input area,
- Local / Global attention selector,
- generated response,
- number of token chunks,
- chunk size,
- attention coverage,
- attention-mask visualization.

The application uses the same core concepts as the notebook.

## Run the GUI

Open Anaconda Prompt:

```bash
conda activate comp5313
```

Then:

```bash
cd "C:\Stayhard\Lakehead_projects\COMP5313_AI\Assignment1_Attention LLM Chatbot"
```

Run:

```bash
streamlit run app.py
```

Streamlit will open the application in the browser.

---

# 21. Notebook and GUI Relationship

The notebook and `app.py` are **not dependent on each other at runtime**.

The relationship is:

```text
                    Project Logic
                         │
             ┌───────────┴───────────┐
             ↓                       ↓
     Jupyter Notebook           Streamlit app.py
             │                       │
      Development +          Interactive GUI
       explanation
```

The notebook documents the development and testing process.

The Python file provides the interactive GUI experience.

The GUI does not need the notebook to be open.

---

# 22. Hardware / Performance Disclaimer

This project was developed and tested using CPU execution.

The original environment reported:

```text
CUDA available: False
```

Therefore, model generation can take longer on lower-end machines.

Performance depends on:

- CPU speed
- available RAM
- background applications
- model size
- input length
- number of generated tokens

The Qwen 1.5B model is practical for local experimentation, but it is still a language model with a significant number of parameters.

### If the model feels slow

Possible options include:

1. Wait for the response to finish.
2. Reduce `max_new_tokens`.
3. Use shorter input.
4. Close unnecessary applications.
5. Use a machine with more RAM or a supported GPU.
6. As a development fallback, use a smaller Qwen model.

The original project intentionally stayed with the 1.5B model because the responses were more useful while still remaining practical on the development laptop.

---

# 23. Internet Requirement

The first time the model is loaded, Hugging Face may need to download the tokenizer and model files.

Therefore:

```text
Internet connection → required for first model download
Local cached model → can be reused after download
```

A Hugging Face authentication warning may appear for unauthenticated requests. For this project, the public model can be downloaded without creating a paid API service.

Download speed depends on the internet connection.

---

# 24. Troubleshooting

## `ModuleNotFoundError`

Activate the correct environment:

```bash
conda activate comp5313
```

Then install the missing package.

Examples:

```bash
pip install transformers
pip install matplotlib
pip install streamlit
pip install jupyter
```

---

## Streamlit does not start

Check:

```bash
streamlit --version
```

If it is missing:

```bash
pip install streamlit
```

Then run:

```bash
streamlit run app.py
```

---

## Model takes time to load

This is normal on CPU.

The model is loaded through:

```python
AutoModelForCausalLM.from_pretrained(...)
```

The first run may also download the model files.

---

## Hugging Face cache / symlink warning on Windows

A Windows warning about symlinks may appear while downloading model files.

This does not automatically mean the model failed to load.

If the notebook reaches:

```text
Model Loaded: Qwen/Qwen2.5-1.5B-Instruct
```

the model loaded successfully.

---

## Jupyter kernel restarted

If the kernel is restarted, variables such as:

```text
model
tokenizer
inputs
token_chunks
processed_chunks
```

are cleared.

Run the notebook again from the beginning so the model and functions are recreated.

---

# 25. Important Development Notes

The notebook intentionally contains some repeated tests.

For example:

- Local mode is tested separately.
- Global mode is tested separately.
- Masks are visualized more than once.
- Q/K/V attention is tested independently.
- The final chatbot pipeline is tested again using long input.

This repetition is intentional for clarity.

The project was developed by testing each part manually before combining the pieces.

The goal was not only:

> "Make a chatbot that works."

The goal was also:

> "Understand what happens between the user's text and the final chatbot response."

That is why the notebook contains intermediate representations, numerical checks and visualizations.

---

# 26. What the Visualizations Show

### Token representation

Shows how the text is represented as token IDs before entering the model.

### Local attention mask

Shows the limited diagonal region where attention is allowed.

### Global attention mask

Shows the full matrix where all token positions can interact.

### Actual attention weights

Shows the result after applying the Local mask and Softmax to the demonstration Q/K/V tensors.

### Local vs Global comparison

Shows the difference between the sparse Local masks and dense Global masks across all chunks of the long test input.

These visualizations make the project easier to understand than only showing the final chatbot response.

---

# 27. Limitations

This is an educational demonstration rather than a production chatbot.

Current limitations include:

1. The Qwen model is running locally and therefore response speed depends on the machine.
2. The model can occasionally generate incorrect or hallucinated information.
3. The custom attention mask is demonstrated at the project/application level and is not directly replacing Qwen's internal Transformer attention mask during final generation.
4. The current chatbot reconstructs the complete processed context before generation.
5. The project does not implement persistent multi-turn conversation memory between separate Streamlit submissions.
6. The Local/Global comparison demonstrates attention connectivity and coverage rather than benchmarking a full production sparse-attention Transformer.

---

# 28. Future Improvements

Possible future improvements include:

- directly injecting custom attention masks into the model's internal Transformer layers,
- persistent multi-turn conversation history,
- configurable chunk size,
- configurable Local window size,
- response-generation controls,
- GPU acceleration,
- token-level attention visualization,
- comparison of generation time between attention strategies,
- larger-context testing,
- additional attention modes.

These are outside the current core implementation but provide a clear path for extending the project.

---

# 29. Final Run Checklist

Before running the project:

```text
[ ] Anaconda / Miniconda installed
[ ] comp5313 environment activated
[ ] Python 3.12 available
[ ] PyTorch installed
[ ] Transformers installed
[ ] Matplotlib installed
[ ] Jupyter installed
[ ] Streamlit installed
[ ] Internet available for first model download
```

Notebook:

```text
[ ] Open Sumit_Assignment1.ipynb
[ ] Run from the beginning
[ ] Confirm Qwen model loads
[ ] Confirm tokenization works
[ ] Confirm 32-token chunking works
[ ] Test Local attention
[ ] Test Global attention
[ ] Check attention coverage
[ ] Run long-context test
[ ] Run final chatbot pipeline
```

GUI:

```text
[ ] Run streamlit run app.py
[ ] Enter a message
[ ] Select Local
[ ] Generate response
[ ] Select Global
[ ] Generate response
[ ] Check attention coverage
[ ] Check attention-mask visualization
```

---

# 30. Final Project Summary

The final project combines a pretrained language model with an understandable attention demonstration layer.

The development process starts from basic model inference, then moves through tokenization, chunking, attention masks, Q/K/V calculations, attention coverage and finally the chatbot pipeline.

The final Streamlit application provides a simple way to interact with the same concept through a GUI.

The main learning outcome is not only the generated response. It is understanding how text can be represented as tokens, divided into chunks, routed through different attention patterns and then passed into a language model for response generation.

---

## Files

```text
Sumit_Assignment1.ipynb   → Main notebook and development record
app.py                    → Streamlit GUI application
ReadMe.pdf                → Project documentation and run handbook
```

**End of README**
