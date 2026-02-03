import boto3
import json
import streamlit as st

# --- CONFIGURATION ---
REGION = "us-west-2"
MODEL_ARN = "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-haiku-20241022-v1:0"
KNOWLEDGE_BASE_ID = "HLH11SEBKY"

model_id_fallback_model = 'anthropic.claude-3-5-haiku-20241022-v1:0'
accept = 'application/json'
contentType = 'application/json'

# Initialize the Bedrock Agent Runtime client.
bedrock_agent_client = boto3.client("bedrock-agent-runtime", region_name=REGION)

# Initialize Bedrock client
bedrock_client = boto3.client(
    service_name='bedrock-runtime', 
    region_name=REGION)

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Dsona Customer Support Bot", page_icon="🛍️", layout="wide")
st.title("🛍️ Dsona Customer Support")
st.markdown("How can I help you today? Ask me anything about Dsona's " \
"products, prices, or policies! or even general questions")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- CHAT INPUT ---
user_input = st.chat_input("What do you have on your mind?")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # --- Step 1: Try Knowledge Base Retrieval + Generation ---
                kb_response = bedrock_agent_client.retrieve_and_generate(
                    input = {
                        "text": user_input
                    },
                    retrieveAndGenerateConfiguration = {
                        "type": "KNOWLEDGE_BASE",
                        "knowledgeBaseConfiguration":{
                            "knowledgeBaseId":KNOWLEDGE_BASE_ID,
                            "modelArn":MODEL_ARN
                        }
                }
                )

                answer = kb_response["output"]["text"]
                citations = kb_response["citations"][0]["retrievedReferences"]
                
                
                # --- Step 2: Determine if Knowledge Base was used ---
                if len(citations) == 0:
                    # No KB data found → fallback to model
                    payload = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 300,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "messages": [{"role": "user", "content": [{"type": "text", "text": user_input}]}]
                    }
                    
                    # Invoke the Claude model
                    model_response = bedrock_client.invoke_model(
                        modelId=model_id_fallback_model,
                        accept=accept,
                        contentType=contentType,
                        body=json.dumps(payload)   # Convert the body to json
                    )

                    # Parse the response body
                    model_output = json.loads(model_response.get('body').read())

                    # Extract the assistant’s text reply
                    answer = model_output["content"][0]["text"]
                    citations = None
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer + "\n\n_(Generated directly by Claude 3.5 Haiku)_"
                    })
                else:
                    # Knowledge Base provided relevant context
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer + "\n\n_(Based on Dsona Knowledge Base)_"
                    })

                    # Add source links if available
                    if citations:
                        source = kb_response["citations"][0]["retrievedReferences"][0]['location']['s3Location']['uri']
    
                        st.session_state["messages"].append({
                            "role": "system",
                            "content": f"Sources:{source}",
                        })

            except Exception as e:
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": f"⚠️ An error occurred: {str(e)}"
                })

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    elif msg["role"] == "system":
        with st.expander("Knowledge Base Sources"):
            st.markdown(msg["content"])