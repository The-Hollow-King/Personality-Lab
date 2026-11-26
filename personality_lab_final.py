import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------
# PAGE CONFIGURATION + DARK THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="The Personality Lab",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
    /* Dark background and readable foreground */
    .stApp {
        background: linear-gradient(160deg, #101827 0%, #151d2e 55%, #111827 100%);
        color: #f3f4f6;
    }
    [data-testid="stHeader"] { background: rgba(16, 24, 39, 0.92); }
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
    [data-testid="stCaptionContainer"] {
        color: #f3f4f6;
    }
    h1 { font-size: 2.65rem !important; color: #f8fafc !important; }
    h2 { font-size: 2.05rem !important; color: #f8fafc !important; }
    h3 { font-size: 1.55rem !important; color: #e2e8f0 !important; }
    p, li, label, .stRadio label, .stMarkdown {
        font-size: 1.12rem !important;
        line-height: 1.65 !important;
    }
    [data-testid="stCaptionContainer"] p { font-size: 0.95rem !important; color: #cbd5e1 !important; }
    div.stButton > button {
        min-height: 3.1rem;
        font-size: 1.08rem !important;
        font-weight: 650;
        border-radius: 12px;
        border: 1px solid #475569;
    }
    div.stButton > button[kind="primary"] {
        background: #7c3aed;
        border-color: #8b5cf6;
        color: white;
    }
    [data-testid="stProgressBar"] > div > div { background-color: #a78bfa; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { background: #1e293b; }
    div[data-baseweb="radio"] { gap: 0.35rem; }
    div[data-baseweb="radio"] label {
        background: rgba(51, 65, 85, 0.42);
        border: 1px solid #475569;
        border-radius: 10px;
        padding: 0.55rem 0.75rem;
        margin: 0.12rem 0;
    }
    [data-testid="stAlert"] { font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 The Personality Lab")
st.write("20 questions. 5 behavioural dimensions. Which profile will you discover?")

# ---------------------------------------------------------
# TRAIN THE CLUSTERING MODEL ON SIMULATED EXAMPLE DATA
# ---------------------------------------------------------
@st.cache_resource
def train_model():
    np.random.seed(42)

    # The five profiles are illustrative patterns used to create demo data.
    # They are not scientific personality types.
    profiles = np.array([
        [4.6, 2.0, 4.5, 3.6],  # Adventure Seekers
        [2.0, 4.6, 3.0, 3.4],  # Master Planners
        [3.5, 3.0, 4.8, 2.0],  # Curious Minds
        [3.0, 3.5, 3.0, 4.8],  # Social Butterflies
        [3.8, 3.8, 3.8, 3.8],  # Balanced Explorers
    ])

    data = []
    for profile in profiles:
        for _ in range(120):
            person = profile + np.random.normal(0, 0.48, 4)
            data.append(np.clip(person, 1, 5))

    df = pd.DataFrame(data, columns=["Risk", "Planning", "Curiosity", "Social"])
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    model = KMeans(n_clusters=5, random_state=42, n_init=15)
    model.fit(X)

    centres = scaler.inverse_transform(model.cluster_centers_)

    names = [
        "The Adventure Seekers",
        "The Master Planners",
        "The Curious Minds",
        "The Social Butterflies",
        "The Balanced Explorers",
    ]
    descriptions = [
        "You enjoy taking chances, embracing uncertainty and discovering what lies beyond the familiar.",
        "You like structure, preparation and having a clear route before taking action.",
        "You are driven by questions, new ideas and the satisfaction of figuring things out.",
        "You get energy from people, collaboration and sharing experiences.",
        "You adapt to different situations, balancing curiosity, preparation, independence and connection.",
    ]

    # Match each learned cluster to one distinct illustrative profile.
    distances = np.linalg.norm(centres[:, None, :] - profiles[None, :, :], axis=2)
    cluster_indices, profile_indices = linear_sum_assignment(distances)
    cluster_names = {}
    cluster_descriptions = {}
    for cluster_index, profile_index in zip(cluster_indices, profile_indices):
        cluster_names[cluster_index] = names[profile_index]
        cluster_descriptions[cluster_index] = descriptions[profile_index]

    return scaler, model, cluster_names, cluster_descriptions


scaler, model, cluster_names, cluster_descriptions = train_model()

# ---------------------------------------------------------
# 20 QUESTIONS: 5 PER DIMENSION
# Scores always run from 1 (lower) to 5 (higher) on that dimension.
# ---------------------------------------------------------
questions = [
    # RISK / COMFORT WITH UNCERTAINTY
    {
        "question": "Your friends invite you to try an activity none of you have done before. What is your reaction?",
        "dimension": "Risk",
        "options": {
            "I'm first in line. Let's try it!": 5,
            "I'm interested; I'll give it a go.": 4,
            "I'll decide after learning a little more.": 3,
            "I'd rather watch someone try it first.": 2,
            "I'd prefer to stick with activities I know.": 1,
        },
    },
    {
        "question": "You have a free afternoon in a city you've never visited. How do you spend it?",
        "dimension": "Risk",
        "options": {
            "Wander wherever the day takes me.": 5,
            "Explore a few places without much planning.": 4,
            "Mix one planned stop with some exploring.": 3,
            "Visit places I have already researched.": 2,
            "Stay somewhere familiar and comfortable.": 1,
        },
    },
    {
        "question": "A project offers a safe, predictable result or a bold idea with an uncertain outcome. Which appeals to you?",
        "dimension": "Risk",
        "options": {
            "The bold idea. A big possibility is worth the uncertainty.": 5,
            "Mostly the bold idea, after a quick check.": 4,
            "I'd weigh both options equally.": 3,
            "I'd lean towards the safer option.": 2,
            "I'd choose the predictable result.": 1,
        },
    },
    {
        "question": "How do you feel about speaking up with an unusual idea that might not work?",
        "dimension": "Risk",
        "options": {
            "I enjoy putting unconventional ideas out there.": 5,
            "I'm usually happy to suggest it.": 4,
            "It depends on the people and situation.": 3,
            "I'd share it only if I felt fairly sure.": 2,
            "I usually keep uncertain ideas to myself.": 1,
        },
    },
    {
        "question": "When plans suddenly change, what best describes you?",
        "dimension": "Risk",
        "options": {
            "Great! I like unexpected turns.": 5,
            "I adapt quickly and see where it goes.": 4,
            "I can manage if I have a moment to adjust.": 3,
            "I feel better when the original plan returns.": 2,
            "Unexpected changes really put me off.": 1,
        },
    },

    # PLANNING / STRUCTURE
    {
        "question": "You have a major assignment due in two weeks. When do you usually begin?",
        "dimension": "Planning",
        "options": {
            "I break it into steps and start early.": 5,
            "I make a plan and begin soon.": 4,
            "I note the deadline and work steadily.": 3,
            "I start once the deadline feels closer.": 2,
            "I usually work in one last-minute push.": 1,
        },
    },
    {
        "question": "How do you organise a day with several things to do?",
        "dimension": "Planning",
        "options": {
            "I schedule tasks and time slots in advance.": 5,
            "I make a clear priority list.": 4,
            "I keep a rough list in my head or on paper.": 3,
            "I choose what to do as the day unfolds.": 2,
            "I rarely organise it beforehand.": 1,
        },
    },
    {
        "question": "Before leaving for a trip, what is your approach?",
        "dimension": "Planning",
        "options": {
            "Everything is checked, packed and planned early.": 5,
            "I prepare a checklist and cover the essentials.": 4,
            "I plan the important parts.": 3,
            "I pack the basics and handle the rest later.": 2,
            "I pack when it's time to leave.": 1,
        },
    },
    {
        "question": "A group project has no assigned roles. What do you do?",
        "dimension": "Planning",
        "options": {
            "I suggest a timeline, roles and clear milestones.": 5,
            "I help divide the work before starting.": 4,
            "I agree on the main tasks, then begin.": 3,
            "I prefer to let the group work it out naturally.": 2,
            "I just pick a task and get started.": 1,
        },
    },
    {
        "question": "How do you keep track of important deadlines or commitments?",
        "dimension": "Planning",
        "options": {
            "A calendar with reminders and regular checks.": 5,
            "A planner or organised to-do list.": 4,
            "A few notes or reminders when needed.": 3,
            "I mostly remember them myself.": 2,
            "I deal with them when they come up.": 1,
        },
    },

    # CURIOSITY / INTEREST IN LEARNING AND EXPLORING IDEAS
    {
        "question": "You come across a device whose purpose you don't recognise. What do you do?",
        "dimension": "Curiosity",
        "options": {
            "Investigate how it works and what it can do.": 5,
            "Look it up and learn the basics.": 4,
            "Explore it if it catches my attention.": 3,
            "Ask someone what it is.": 2,
            "Move on without giving it much thought.": 1,
        },
    },
    {
        "question": "Which weekend activity sounds most satisfying?",
        "dimension": "Curiosity",
        "options": {
            "Learning a completely unfamiliar skill or subject.": 5,
            "Visiting a place or exhibition I've never seen.": 4,
            "Trying a new version of a familiar hobby.": 3,
            "Revisiting a favourite film, game or book.": 2,
            "Doing my usual routine.": 1,
        },
    },
    {
        "question": "During a conversation, someone mentions an idea you've never heard of. What happens next?",
        "dimension": "Curiosity",
        "options": {
            "I ask questions and want to explore it deeply.": 5,
            "I look for more details and examples.": 4,
            "I listen and see if it interests me.": 3,
            "I follow along but don't usually investigate further.": 2,
            "I quickly move the conversation to another topic.": 1,
        },
    },
    {
        "question": "When you find out that something you believed may be incorrect, what do you do?",
        "dimension": "Curiosity",
        "options": {
            "I investigate the evidence and rethink my view.": 5,
            "I look into the alternative explanation.": 4,
            "I consider it and keep an open mind.": 3,
            "I wait until I hear more about it.": 2,
            "I usually stick with what I already know.": 1,
        },
    },
    {
        "question": "What kind of online article or video are you most likely to open?",
        "dimension": "Curiosity",
        "options": {
            "A deep dive explaining an unfamiliar topic.": 5,
            "A surprising discovery or how-things-work story.": 4,
            "Something new related to my interests.": 3,
            "A familiar topic I already enjoy.": 2,
            "The same kind of content I usually watch.": 1,
        },
    },

    # SOCIAL / ENJOYMENT OF INTERACTION AND COLLABORATION
    {
        "question": "At an event where you know very few people, what are you most likely to do?",
        "dimension": "Social",
        "options": {
            "Introduce myself to lots of people.": 5,
            "Start conversations with a few new people.": 4,
            "Talk to people when a natural opportunity appears.": 3,
            "Stay close to the people I already know.": 2,
            "Keep to myself for most of the event.": 1,
        },
    },
    {
        "question": "You can choose between working alone or with a group on an interesting task. What do you prefer?",
        "dimension": "Social",
        "options": {
            "A group where everyone shares ideas.": 5,
            "Mostly teamwork, with some individual time.": 4,
            "Either works equally well for me.": 3,
            "Mostly independent work.": 2,
            "Working completely on my own.": 1,
        },
    },
    {
        "question": "When something exciting happens to you, what is your first instinct?",
        "dimension": "Social",
        "options": {
            "Tell people and celebrate together.": 5,
            "Share it with several friends.": 4,
            "Tell a few people I'm close to.": 3,
            "Mention it if someone asks.": 2,
            "Keep it mostly to myself.": 1,
        },
    },
    {
        "question": "A friend is organising a group activity. How involved would you like to be?",
        "dimension": "Social",
        "options": {
            "I'd love to help organise and bring everyone together.": 5,
            "I'd happily help and participate.": 4,
            "I'll join in and help if needed.": 3,
            "I'd rather just attend.": 2,
            "I'd prefer to do something independently.": 1,
        },
    },
    {
        "question": "In a discussion, what role do you naturally take?",
        "dimension": "Social",
        "options": {
            "I keep the conversation moving and involve everyone.": 5,
            "I contribute often and build on others' ideas.": 4,
            "I contribute when I have something to add.": 3,
            "I mostly listen and speak occasionally.": 2,
            "I prefer listening rather than joining in.": 1,
        },
    },
]

# ---------------------------------------------------------
# SESSION STATE: ONE QUESTION AT A TIME
# ---------------------------------------------------------
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "quiz_completed" not in st.session_state:
    st.session_state.quiz_completed = False

if not st.session_state.quiz_completed:
    st.subheader("The Personality Experiment")
    st.write("Choose the answer that feels most like you. There are no right or wrong answers.")

    i = st.session_state.current_question
    total_questions = len(questions)
    current = questions[i]

    st.progress((i + 1) / total_questions)
    st.caption(f"Question {i + 1} of {total_questions}")
    st.subheader(current["question"])

    selected_answer = st.radio(
        "Choose one option:",
        list(current["options"].keys()),
        index=None,
        key=f"answer_{i}",
    )

    previous_col, next_col = st.columns(2)
    with previous_col:
        if st.button("← Previous", disabled=(i == 0), use_container_width=True):
            if selected_answer is not None:
                st.session_state.answers[i] = (
                    current["dimension"], current["options"][selected_answer]
                )
            st.session_state.current_question -= 1
            st.rerun()

    with next_col:
        is_last_question = i == total_questions - 1
        next_label = "See My Results →" if is_last_question else "Next →"
        if st.button(
            next_label,
            disabled=(selected_answer is None),
            type="primary",
            use_container_width=True,
        ):
            st.session_state.answers[i] = (
                current["dimension"], current["options"][selected_answer]
            )
            if is_last_question:
                st.session_state.quiz_completed = True
            else:
                st.session_state.current_question += 1
            st.rerun()

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------
if st.session_state.quiz_completed:
    scores = {"Risk": [], "Planning": [], "Curiosity": [], "Social": []}
    for dimension, score in st.session_state.answers.values():
        scores[dimension].append(score)

    user_data = pd.DataFrame([{
        dimension: float(np.mean(values)) for dimension, values in scores.items()
    }])

    prediction = model.predict(scaler.transform(user_data))[0]
    personality = cluster_names[prediction]
    description = cluster_descriptions[prediction]

    fun_facts = {
        "The Adventure Seekers": [
            "Your imaginary passport probably has more stamps than blank pages.",
            "You might be the person who says, 'Let's find out!' before checking the instructions.",
            "A change of plans can turn into the best part of your day.",
        ],
        "The Master Planners": [
            "You probably appreciate a good checklist more than most people appreciate a surprise.",
            "You might plan your route and still check it one more time.",
            "For you, being prepared can make room for better adventures later.",
        ],
        "The Curious Minds": [
            "One small question can send you down a wonderfully long research rabbit hole.",
            "You might read the explanation after the explanation just to understand the details.",
            "Your browser tabs may be a museum of things you wanted to learn.",
        ],
        "The Social Butterflies": [
            "You might turn a quick hello into a surprisingly long conversation.",
            "Group plans are often more fun when everyone gets involved.",
            "You may be the unofficial connector in a room full of people.",
        ],
        "The Balanced Explorers": [
            "You can appreciate a plan and still leave room for a little spontaneity.",
            "You may be comfortable switching between solo focus and group energy.",
            "You like having options rather than being locked into just one way of doing things.",
        ],
    }

    st.divider()
    st.balloons()
    st.header("Your Personality Result")
    st.success(personality)
    st.write(description)

    st.subheader("Your Behavioural Profile")
    st.bar_chart(user_data.T, y_label="Average score (1–5)")

    st.subheader("Your scores")
    score_table = user_data.T.rename(columns={0: "Your average score"})
    st.dataframe(score_table, use_container_width=True)

    st.subheader("🎲 A few fun facts about your profile")
    chosen_facts = np.random.default_rng().choice(
        fun_facts[personality], size=2, replace=False
    )
    for fact in chosen_facts:
        st.markdown(f"- {fact}")

    st.caption(
        "This is an educational demonstration of clustering using simulated data. "
        "The profile names and descriptions are playful labels, not scientifically "
        "validated personality types or psychological assessments."
    )

    if st.button("↻ Take the experiment again", use_container_width=True):
        st.session_state.current_question = 0
        st.session_state.answers = {}
        st.session_state.quiz_completed = False
        for question_index in range(len(questions)):
            widget_key = f"answer_{question_index}"
            if widget_key in st.session_state:
                del st.session_state[widget_key]
        st.rerun()
