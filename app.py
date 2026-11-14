import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. PAGE CONFIGURATION
# ----------------------------

st.set_page_config(
    page_title="Personality Lab",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 The Personality Lab")
st.write("Discover which personality group you belong to!")

# ----------------------------
# 2. CREATE SAMPLE DATA
# ----------------------------

@st.cache_resource
def train_model():

    np.random.seed(42)

    # 21 4 Simulated behavioural groups
    profiles = np.array([
        [4.5, 2.0, 4.5, 3.5],  # Adventurous
        [2.0, 4.5, 3.0, 3.5],  # Planners
        [3.5, 3.0, 4.8, 2.0],  # Curious
        [3.0, 3.5, 3.0, 4.8],  # Social
    ])

    data = []

    for profile in profiles:
        for _ in range(100):
            person = profile + np.random.normal(
                0, 0.55, 4
            )
            person = np.clip(person, 1, 5)
            data.append(person)

    df = pd.DataFrame(
        data,
        columns=[
            "Risk",
            "Planning",
            "Curiosity",
            "Social"
        ]
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    model.fit(X)

    # Give clusters readable names by comparing
    # their centres with our simulated profiles.
    centres = scaler.inverse_transform(
        model.cluster_centers_
    )

    names = [
        "The Adventure Seekers",
        "The Master Planners",
        "The Curious Minds",
        "The Social Butterflies"
    ]

    descriptions = [
        "You love taking chances and exploring new possibilities!",
        "You prefer structure, preparation and thoughtful decisions.",
        "You are naturally curious and love discovering new things.",
        "You enjoy people, collaboration and shared experiences."
    ]

    anchors = profiles

    cluster_names = {}
    cluster_descriptions = {}

    for i, centre in enumerate(centres):
        closest = np.argmin(
            np.linalg.norm(anchors - centre, axis=1)
        )

        cluster_names[i] = names[closest]
        cluster_descriptions[i] = descriptions[closest]

    return scaler, model, cluster_names, cluster_descriptions


scaler, model, cluster_names, cluster_descriptions = train_model()

# ----------------------------
# 3. QUESTIONNAIRE
# ----------------------------

questions = [
    {
        "question": "You are offered a chance to try something completely new. What do you do?",
        "dimension": "Risk",
        "options": {
            "Absolutely! I love new experiences.": 5,
            "I'll probably give it a try.": 4,
            "Depends on the situation.": 3,
            "I'd rather stick to what I know.": 2,
            "No thanks!": 1
        }
    },
    {
        "question": "Would you go on a spontaneous adventure?",
        "dimension": "Risk",
        "options": {
            "Definitely!": 5,
            "Probably.": 4,
            "Maybe.": 3,
            "Only with some planning.": 2,
            "Not really.": 1
        }
    },
    {
        "question": "How do you approach an important task?",
        "dimension": "Planning",
        "options": {
            "I plan every detail beforehand.": 5,
            "I make a proper schedule.": 4,
            "I plan the important parts.": 3,
            "I figure things out along the way.": 2,
            "I just wing it!": 1
        }
    },
    {
        "question": "Your friends suggest a last-minute trip. Your reaction?",
        "dimension": "Planning",
        "options": {
            "I need an itinerary first!": 5,
            "Let me check everything.": 4,
            "I'll think about it.": 3,
            "Sounds fun, let's go.": 2,
            "Pack your bags. We're leaving!": 1
        }
    },
    {
        "question": "You encounter something mysterious. What do you do?",
        "dimension": "Curiosity",
        "options": {
            "Investigate everything about it.": 5,
            "Try to understand how it works.": 4,
            "Look into it if it's interesting.": 3,
            "Maybe search for an explanation.": 2,
            "Ignore it.": 1
        }
    },
    {
        "question": "Which activity sounds most appealing?",
        "dimension": "Curiosity",
        "options": {
            "Learning something completely new.": 5,
            "Exploring an unfamiliar place.": 4,
            "Trying a new hobby.": 3,
            "Watching something entertaining.": 2,
            "Doing something familiar.": 1
        }
    },
    {
        "question": "At a party, where would you most likely be?",
        "dimension": "Social",
        "options": {
            "Talking to absolutely everyone!": 5,
            "Meeting lots of new people.": 4,
            "Chatting with a few people.": 3,
            "Sticking with my friends.": 2,
            "Finding a quiet corner.": 1
        }
    },
    {
        "question": "How do you prefer working on a project?",
        "dimension": "Social",
        "options": {
            "Definitely as a team!": 5,
            "With a group of friends.": 4,
            "A mixture of both.": 3,
            "Mostly independently.": 2,
            "Completely on my own.": 1
        }
    }
]

# ----------------------------
# 4. COLLECT ANSWERS
# ----------------------------

st.subheader("Answer the questions!")

st.write("There are 8 questions. Choose the answer that suits you best.")

with st.form("personality_form"):

    answers = {}

    for i, q in enumerate(questions):

        answer = st.radio(
            f"{i+1}. {q['question']}",
            list(q["options"].keys()),
            key=f"question_{i}"
        )

        answers[q["dimension"]] = (
            answers.get(q["dimension"], []) +
            [q["options"][answer]]
        )

    submitted = st.form_submit_button(
        "Discover My Personality!"
    )

# ----------------------------
# 5. ANALYSE RESPONSES
# ----------------------------

if submitted:

    # Average the two answers for each dimension
    user_data = pd.DataFrame([{
        dimension: np.mean(scores)
        for dimension, scores in answers.items()
    }])

    user_scaled = scaler.transform(user_data)

    prediction = model.predict(user_scaled)[0]

    personality = cluster_names[prediction]
    description = cluster_descriptions[prediction]

    # ------------------------
    # 6. DISPLAY RESULTS
    # ------------------------

    st.divider()

    st.balloons()

    st.header("Your Personality Result!")

    st.success(personality)

    st.write(description)

    st.subheader("Your Behavioural Profile")

    st.bar_chart(
        user_data.T,
        y_label="Score (1–5)"
    )

    st.write("### Your answers")

    st.dataframe(
        user_data.T.rename(
            columns={0: "Your Score"}
        )
    )

    st.caption(
        "This is an experimental clustering demo, "
        "not a scientifically validated personality test."
    )
