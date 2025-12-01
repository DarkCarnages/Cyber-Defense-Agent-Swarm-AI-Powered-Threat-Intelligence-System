import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
from crewai import Agent, Crew
from groq import Groq
from dotenv import load_dotenv


load_dotenv()

MODEL_PATH = r"D:\5th sem\Cyber CP and HA\NEW\Models\random_forest_cyber_classifier.pkl"
TEST_DATA_PATH = r"D:\5th sem\Cyber CP and HA\NEW\Dataset\Testing.csv"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))  #LLM model loaded 

VALID_PROTOCOLS = [
    'udp','arp','tcp','igmp','ospf','sctp','gre','ggp','ip','ipnip','st2','argus','chaos','egp',
    'emcon','nvp','pup','xnet','mux','dcn','hmp','prm','trunk-1','trunk-2','xns-idp','leaf-1',
    'leaf-2','irtp','rdp','netblt','mfe-nsp','merit-inp','3pc','idpr','ddp','idpr-cmtp','tp++',
    'ipv6','sdrp','ipv6-frag','ipv6-route','idrp','mhrp','i-nlsp','rvd','mobile','narp','skip',
    'tlsp','ipv6-no','any','ipv6-opts','cftp','sat-expak','ippc','kryptolan','sat-mon','cpnx',
    'wsn','pvp','br-sat-mon','sun-nd','wb-mon','vmtp','ttp','vines','nsfnet-igp','dgp','eigrp',
    'tcf','sprite-rpc','larp','mtp','ax.25','ipip','aes-sp3-d','micp','encap','pri-enc','gmtp',
    'ifmp','pnni','qnx','scps','cbt','bbn-rcc','igp','bna','swipe','visa','ipcv','cphb','iso-tp4',
    'wb-expak','sep','secure-vmtp','xtp','il','rsvp','unas','fc','iso-ip','etherip','pim','aris',
    'a/n','ipcomp','snp','compaq-peer','ipx-n-ip','pgm','vrrp','l2tp','zero','ddx','iatp','stp',
    'srp','uti','sm','smp','isis','ptp','fire','crtp','crudp','sccopmce','iplt','pipe','sps','ib'
]

@st.cache_resource
def load_resources():
    model_data = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_DATA_PATH).reset_index(drop=True)
    return model_data["model"], model_data["label_encoder"], df


def groq_summary(prediction, protocol, features):
    if not os.environ.get("GROQ_API_KEY"):
        return {
            "description": "Groq API key missing.",
            "how_it_works": "",
            "mitigation": []
        }

    prompt = f"""
You are a cybersecurity expert. Explain this threat clearly.

Attack type: {prediction}
Protocol used: {protocol.upper()}
Key suspicious indicators: {', '.join(features)}

Respond in JSON with:

- description (2-4 sentences)
- how_it_works (2-4 sentences)
- mitigation (5 bullet prevention steps)
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are an expert SOC analyst."},
            {"role": "user", "content": prompt}
        ]
    )

    return json.loads(response.choices[0].message.content)


llm = client  

Watcher_Agent = Agent(
    role='Watcher',
    goal='Continuously monitor incoming network flow data and feed it into the Analyst Agent for prediction.',
    backstory=(
        "You are the Watcher Agent — a vigilant observer of all network activity. "
        "You detect new logs or packets arriving from sensors or datasets. "
        "Your task is to parse and forward these observations to the Analyst Agent."
    ),
    verbose=True,
    allow_delegation=True,
    llm=llm
)

Analyst_Agent = Agent(
    role='Analyst',
    goal='Use the pre-trained classical ML model to classify incoming samples as normal or anomalous.',
    backstory=(
        "You are the Analyst Agent — the intelligence behind the cyber defense system. "
        "You use a trained ML model to analyze each incoming event and predict whether it represents an attack or benign activity. "
        "You then pass this result to the Remediator Agent."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

Remediator_Agent = Agent(
    role='Remediator',
    goal='Take corrective actions when an anomaly is detected, such as isolating nodes or alerting admins.',
    backstory=(
        "You are the Remediator Agent — the protector of the network. "
        "When the Analyst flags a potential threat, you take swift action. "
        "Depending on severity, you can isolate infected nodes, update firewall rules, or trigger alerts. "
        "All actions are logged for transparency."
    ),
    verbose=True,
    allow_delegation=True,
    llm=llm
)

Explainer_Agent = Agent(
    role='Explainer',
    goal='Generate explainable insights for each detection using SHAP, helping analysts understand model reasoning.',
    backstory=(
        "You are the Explainer Agent — a transparent communicator between AI models and human analysts. "
        "Your job is to clarify why a sample was marked as a threat. "
        "You highlight key features like SYNFlag Count or Flow Duration that influenced the model’s decision."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

crew = Crew(agents=[Watcher_Agent, Analyst_Agent, Remediator_Agent, Explainer_Agent])


def analyze_event(new_data: pd.DataFrame, model, label_encoder):
 
    print("Watcher: received new event.")
    features = new_data.copy()

    if hasattr(model, "feature_names_in_"):
        features = features[model.feature_names_in_]

    pred = model.predict(features)
    decoded = label_encoder.inverse_transform(pred)[0]
    print("Analyst: prediction =", decoded)

    if decoded != "Normal":
        remediation = f"Alert: {decoded} attack detected. Executing mitigation and isolation protocol."
        is_threat = True
    else:
        remediation = "No threat detected. Event classified as normal. Continue monitoring."
        is_threat = False

    print("Remediator:", remediation)

   
    if "proto" in features.columns:
        proto = str(features.iloc[0]["proto"]).lower()
        proto = proto if proto in VALID_PROTOCOLS else "UNKNOWN PROTOCOL"
    else:
        proto = "UNKNOWN PROTOCOL"

    preprocess = model.named_steps['preprocess']
    inner_model = model.named_steps['model']

    transformed = preprocess.transform(features)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = transformed.astype(float)

    explainer = shap.TreeExplainer(inner_model, feature_perturbation="interventional")
    shap_values = explainer.shap_values(transformed, check_additivity=False)

    if isinstance(shap_values, list):
        shap_values = shap_values[np.argmax(inner_model.predict_proba(transformed)[0])]

    ohe_cols = preprocess.named_transformers_['onehot'].get_feature_names_out(
        ['proto', 'service', 'state']
    )
    remaining_cols = [c for c in features.columns if c not in ['proto', 'service', 'state']]
    final_feature_names = list(ohe_cols) + remaining_cols

    importance_scores = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(importance_scores)[::-1][:5]
    top_indices = np.array(top_indices).flatten().tolist()

    top_features = [final_feature_names[i] for i in top_indices]

    explanation = f"Top influencing features: {top_features}"
    print("Explainer:", explanation)

 
    llm_report = None
    if is_threat:
        simple_key_features = ["rate", "sbytes", "dbytes"]
        llm_report = groq_summary(decoded, proto, simple_key_features)

    return {
        "prediction": decoded,
        "protocol": proto.upper(),
        "remediation": remediation,
        "explanation": explanation,
        "llm": llm_report,
        "is_threat": is_threat
    }



st.set_page_config(page_title="Cyber Defense AI Agent Swarm", layout="wide")

st.title("Cyber Defense Agent Swarm - Threat Intelligence UI")

model, encoder, df = load_resources()

st.sidebar.header("Sample Selector")
row_id = st.sidebar.number_input("Pick a row from dataset", 0, len(df) - 1, 0)
analyze_btn = st.sidebar.button("Analyze")

if analyze_btn:
 
    sample = df.iloc[[row_id]].drop(["attack_cat", "label"], axis=1)

    result = analyze_event(sample, model, encoder)

    st.subheader("Threat Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Class", result["prediction"])
    with col2:
        st.metric("Protocol", result["protocol"])

    st.markdown("---")

    st.subheader("Model Explanation")
    st.info(result["explanation"])

    st.subheader("Agent Remediation Decision")
    if result["is_threat"]:
        st.error(result["remediation"])
    else:
        st.success(result["remediation"])

    
    if result["is_threat"] and result["llm"] is not None:
        st.subheader("AI Security Report")
        st.write(f"Description: {result['llm'].get('description', '')}")
        st.write(f"How It Works: {result['llm'].get('how_it_works', '')}")

        st.write("Mitigation Steps:")
        for step in result["llm"].get("mitigation", []):
            st.markdown(f"- {step}")
    else:
        st.subheader("AI Security Report")
        st.write("No additional threat intelligence generated for normal traffic classification.")

else:
    st.info("Select a row and click Analyze to run the agent swarm.")
