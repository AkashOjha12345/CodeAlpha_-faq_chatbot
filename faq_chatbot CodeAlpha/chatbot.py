from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faq_data import faqs

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(questions)

def get_response(user_input):

    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, question_vectors)

    best_match_index = similarity.argmax()

    score = similarity[0, best_match_index]

    if score < 0.2:
        return "Sorry, I don't understand your question."

    return answers[best_match_index]