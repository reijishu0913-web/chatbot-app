from flask import Flask, request, render_template
import os, requests

app = Flask(__name__)

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # Renderで設定する
MODEL_URL = "https://api-inference.huggingface.co/models/rinna/japanese-gpt2-small"

def generate_text(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 80,
            "temperature": 0.8,
            "top_p": 0.9,
            "do_sample": True,
        }
    }
    r = requests.post(MODEL_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # 出力形式に合わせてテキストを取り出す
    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
        return data[0]["generated_text"]
    # pipeline形式の返り値対策
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    return str(data)

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form.get("message", "")
        try:
            response = generate_text(user_input)
        except Exception as e:
            response = f"エラー: {e}"
        return render_template("index.html", user_input=user_input, response=response)
    return render_template("index.html")

if __name__ == "__main__":
    # ローカル実行用（Renderではgunicornが起動）
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
