# 🧠 The Personality Lab

An interactive Streamlit app that demonstrates **unsupervised machine learning (K-Means clustering)** through a playful 20-question personality quiz. Answer the questions, and the app places you in one of five behavioural profiles based on how your answers compare to a trained clustering model.

> ⚠️ **Disclaimer:** This is an educational demo built on simulated data. The profile names and descriptions are playful labels, not scientifically validated personality types or psychological assessments.

## ✨ Features

- **20-question quiz**, shown one question at a time with a progress bar
- **Back and forward navigation**, so you can revisit and change earlier answers
- **Four behavioural dimensions**, with five questions each, scored 1–5:
  - **Risk**: comfort with uncertainty and trying new things
  - **Planning**: preference for structure and preparation
  - **Curiosity**: interest in learning and exploring ideas
  - **Social**: enjoyment of interaction and collaboration
- **Five profiles**: The Adventure Seekers, The Master Planners, The Curious Minds, The Social Butterflies and The Balanced Explorers
- **Results page** with your assigned profile, a bar chart of your dimension scores, a score table and randomly picked fun facts
- **Dark-themed UI** with custom CSS styling
- **Retake option** that resets the quiz

## ⚙️ How It Works

1. **Simulated training data:** 600 synthetic respondents (120 per profile) are generated around five illustrative score patterns, with Gaussian noise added and values clipped to the 1–5 range.
2. **Clustering:** The data is standardised with `StandardScaler` and clustered using `KMeans` (k = 5).
3. **Cluster labelling:** Each learned cluster is matched to exactly one named profile using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), based on distance to the illustrative profile centres.
4. **Prediction:** Your answers are averaged per dimension, scaled with the same scaler and assigned to the nearest cluster.

The model is trained once on startup and cached with `@st.cache_resource`.

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/): UI and session state
- [scikit-learn](https://scikit-learn.org/): `KMeans`, `StandardScaler`
- [SciPy](https://scipy.org/): `linear_sum_assignment`
- [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/): data handling

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install streamlit pandas numpy scikit-learn scipy

# Run the app
streamlit run personality_lab_final.py
```

## 📚 What You Can Learn From It

- How K-Means clustering groups data points
- Why feature scaling matters before clustering
- How to map unlabelled clusters to meaningful names
- How to build a multi-step interactive app with Streamlit session state
