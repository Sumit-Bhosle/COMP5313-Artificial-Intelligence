import streamlit as st
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(
    page_title="Adjustable Attention Chatbot",
    page_icon=None,
    layout="wide"
)

@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_model()

def create_token_chunks(text, chunk_size=32):
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = [
        tokens[i:i + chunk_size]
        for i in range(0, len(tokens), chunk_size)
    ]

    return chunks

def create_attention_mask(sequence_len, mode="local", window_size=2):

    if mode == "local":
        mask = torch.zeros(
            (sequence_len, sequence_len),
            dtype=torch.int
        )

        for i in range(sequence_len):
            for j in range(sequence_len):
                if abs(i - j) <= window_size:
                    mask[i, j] = 1

    elif mode == "global":
        mask = torch.ones(
            (sequence_len, sequence_len),
            dtype=torch.int
        )

    else:
        raise ValueError("Mode must be 'local' or 'global'")

    return mask

def process_attention(chunks, mode="local", window_size=2):

    processed_chunks = []

    for chunk in chunks:
        sequence_len = len(chunk)

        mask = create_attention_mask(
            sequence_len=sequence_len,
            mode=mode,
            window_size=window_size
        )

        processed_chunks.append({
            "tokens": chunk,
            "mask": mask
        })

    return processed_chunks

def generate_chatbot_response(user_text, mode="local", chunk_size=32):

    # 1. Split input into token chunks
    chunks = create_token_chunks(
        user_text,
        chunk_size=chunk_size
    )

    # 2. Apply selected attention mode
    processed = process_attention(
        chunks,
        mode=mode,
        window_size=2
    )

    # Calculate attention coverage
    total_allowed = 0
    total_possible = 0

    for item in processed:
        mask = item["mask"]

        total_allowed += mask.sum().item()
        total_possible += mask.numel()

    attention_coverage = (
        total_allowed / total_possible
    ) * 100
    # Use the first chunk for visualization
    attention_mask = processed[0]["mask"]

    # 3. Reconstruct the complete context
    context_parts = []

    for chunk in processed:
        text = tokenizer.decode(
            chunk["tokens"],
            skip_special_tokens=True
        )
        context_parts.append(text)

    context = " ".join(context_parts)

    # 4. Create chat prompt
    messages = [
        {
            "role": "user",
            "content": context
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    # 5. Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return response, len(chunks), attention_coverage, attention_mask


st.title("Adjustable Attention Chatbot")

st.write(
    "A Qwen-based chatbot with configurable local and global attention."
)

st.sidebar.header("Attention Control")

attention_mode = st.sidebar.radio(
    "Select Attention Mode",
    ["local", "global"]
)

st.sidebar.write(f"Selected mode: **{attention_mode.upper()}**")

st.divider()

st.subheader("Chat")

user_input = st.text_area(
    "Enter your message:",
    height=150,
    placeholder="Ask something..."
)

send = st.button("Send")

if send:
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Generating response..."):
            response, chunk_count, attention_coverage, attention_mask = generate_chatbot_response(
                user_input,
                mode=attention_mode
            )

        st.subheader("Assistant")
        st.write(response)

        st.caption(
            f"Attention Mode: {attention_mode.upper()} "
            f"• Token Chunks: {chunk_count} "
            f"• Chunk Size: 32 "
            f"• Attention Coverage: {attention_coverage:.2f}%"
        )

        st.subheader("Attention Mask")

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.imshow(attention_mask)

        ax.set_xlabel("Key Token Position")
        ax.set_ylabel("Query Token Position")
        ax.set_title(
            f"{attention_mode.upper()} Attention Mask"
        )

        st.pyplot(fig, width="content")
        plt.close(fig)

