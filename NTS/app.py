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
        url = data.get('url')
        text = data.get('text')

        if url:
            news_text = extract_news(url)
        elif text:
            news_text = text
        summarized_text = summarize(news_text)
        return jsonify({'summary': summarized_text})
    except Exception as e:
        #return jsonify({'summary': str(e)})
        return jsonify({'summary':"Cannot Extract content from the url"})
if __name__ == '__main__':
    app.run()
