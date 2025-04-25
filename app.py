from flask import Flask, request, jsonify
from EmotionalStoryChatbot import EmotionalStoryChatbot

app = Flask(__name__)
chatbot = EmotionalStoryChatbot()

@app.route("/")
def home():
    return "Emotional Story Chatbot is running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    
    if user_input.lower() == "summarize":
        summary = chatbot.summarize_conversation()
        return jsonify({"type": "summary", "response": summary})
    
    emotion, emotion_score = chatbot.detect_emotion(user_input)
    response = chatbot.generate_response(user_input, emotion, emotion_score)
    
    return jsonify({
        "type": "chat",
        "emotion": emotion,
        "emotion_score": round(emotion_score, 2),
        "response": response
    })

if __name__ == "__main__":
    app.run(debug=True)
