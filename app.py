from flask import Flask, request, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer

# Flaskアプリを作成
app = Flask(__name__)

# 日本語GPTモデルを読み込む
model_name = "rinna/japanese-gpt2-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# "/" にアクセスしたときの処理
@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form["message"]  # 入力フォームからメッセージ取得
        inputs = tokenizer.encode(user_input, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_length=100,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return render_template("index.html", user_input=user_input, response=response)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

