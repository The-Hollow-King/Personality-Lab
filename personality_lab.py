import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page configuration

st.set_page_config(
    page_title="Personality Lab",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 The Personality Lab")
st.write("Discover which personality group you belong to!")

# 2. Creating sample data

@st.cache_resource
def train_model():

    np.random.seed(42)

    # 4 Simulated behavioural groups
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

    # Give clusters readable names by comparing their centres with our simulated profiles.
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

# Questions

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

# Collecting answers: one question at a time

# Session state preserves the visitor's progress when Streamlit reruns the app.
if "current_question" not in st.session_state:
    st.session_state.current_question = 0

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "quiz_completed" not in st.session_state:
    st.session_state.quiz_completed = False


if not st.session_state.quiz_completed:
    st.subheader("The Personality Experiment")
    st.write("Answer each question to discover your behavioural profile.")

    # Which question should be displayed?
    i = st.session_state.current_question
    total_questions = len(questions)
    current = questions[i]

    # Progress indicator
    st.progress((i + 1) / total_questions)
    st.caption(f"Question {i + 1} of {total_questions}")

    st.subheader(current["question"])

    # Show only the current question's answer choices.
    selected_answer = st.radio(
        "Choose the answer that suits you best:",
        list(current["options"].keys()),
        index=None,
        key=f"answer_{i}"
    )

    previous_col, next_col = st.columns(2)

    with previous_col:
        if st.button("← Previous", disabled=(i == 0), use_container_width=True):
            # Preserve the current answer before moving backwards.
            if selected_answer is not None:
                st.session_state.answers[i] = (
                    current["dimension"],
                    current["options"][selected_answer]
                )

            st.session_state.current_question -= 1
            st.rerun()

    with next_col:
        is_last_question = (i == total_questions - 1)
        next_label = "See My Results →" if is_last_question else "Next →"

        if st.button(
            next_label,
            disabled=(selected_answer is None),
            type="primary",
            use_container_width=True
        ):
            # Store this answer as (dimension, numerical score).
            st.session_state.answers[i] = (
                current["dimension"],
                current["options"][selected_answer]
            )

            if is_last_question:
                st.session_state.quiz_completed = True
            else:
                st.session_state.current_question += 1

            st.rerun()


# Analysing outputs after all eight questions have been answered
if st.session_state.quiz_completed:

    # Group the two answers belonging to each behavioural dimension.
    scores = {
        "Risk": [],
        "Planning": [],
        "Curiosity": [],
        "Social": []
    }

    for dimension, score in st.session_state.answers.values():
        scores[dimension].append(score)

    # Average the two answers for each dimension.
    user_data = pd.DataFrame([{
        dimension: np.mean(values)
        for dimension, values in scores.items()
    }])

    # Use the same scaler and trained K-Means model as before.
    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)[0]

    personality = cluster_names[prediction]
    description = cluster_descriptions[prediction]

    # Display the result.
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

    st.write("### Your scores")
    st.dataframe(
        user_data.T.rename(columns={0: "Your Score"}),
        use_container_width=True
    )

    st.caption(
        "This is an experimental clustering demo, "
        "not a scientifically validated personality test."
    )

    # Allow another visitor to take the quiz on the same device.
    if st.button("Take the experiment again", use_container_width=True):
        st.session_state.current_question = 0
        st.session_state.answers = {}
        st.session_state.quiz_completed = False

        # Clear saved radio selections so the next visitor starts fresh.
        for question_index in range(len(questions)):
            widget_key = f"answer_{question_index}"
            if widget_key in st.session_state:
                del st.session_state[widget_key]

        st.rerun()
