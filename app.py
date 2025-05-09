from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

def is_relevant_job(user_title, job_title, threshold=0.6):
    embeddings = model.encode([user_title, job_title], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity > threshold

def get_relevant_jobs(user_title):
    headers = {"User-Agent": "Mozilla/5.0"}
    job_results = []

    for page in range(0, 3):
        search_url = f"https://wuzzuf.net/search/jobs/?q={user_title.replace(' ', '-')}&start={page * 10}"
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        job_cards = soup.find_all("div", {"class": "css-1gatmva"})
        for card in job_cards:
            title_element = card.find("h2", {"class": "css-m604qf"})
            if not title_element:
                continue

            job_name = title_element.text.strip()
            link_tag = title_element.find("a", href=True)

            if link_tag:
                full_link = urljoin("https://wuzzuf.net", link_tag['href'])
                if is_relevant_job(user_title, job_name):
                    job_results.append({"title": job_name, "link": full_link})

    return job_results

@app.route("/get-jobs", methods=["POST"])
def job_api():
    data = request.json
    user_title = data.get("job_title", "")
    if not user_title:
        return jsonify({"error": "job_title is required"}), 400

    jobs = get_relevant_jobs(user_title)
    return jsonify({"jobs": jobs})

if __name__ == "__main__":
    app.run(debug=True)
