from flask import Flask, request, render_template, jsonify
from urlToNewsReport import extract_news
from summary import summarize

app = Flask(__name__)
app.config.update(
    TEMPLATES_AUTO_RELOAD=True
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/setNewsText', methods=['POST'])
def setNewsText():
    try:
        data = request.get_json()
        url = data.get('url')  # Get the URL from the request
        text = data.get('text')  # Get the entered text
        if url:  # If URL is provided, extract news content
            news_text = extract_news(url)
        elif text:  # If text is provided, use it as news content
            news_text = text
        else:
            return jsonify({'summary': 'No input received'})

        summarized_text = summarize(news_text)
        if summarized_text:
            return jsonify({'summary': summarized_text})
        else:
            return jsonify({'summary': 'Failed to fetch news content'})
    except Exception as e:
        return jsonify({'summary': str(e)})

if __name__ == '__main__':
    app.run()
