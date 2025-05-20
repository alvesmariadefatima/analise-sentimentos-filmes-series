from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)
CORS(app)

# ✅ Carregar modelo e vetorizador
modelo_path = "modelo_sentimentos.pkl"
vetorizador_path = "vetorizador.pkl"

if os.path.exists(modelo_path) and os.path.exists(vetorizador_path):
    modelo = joblib.load(modelo_path)
    vetorizador = joblib.load(vetorizador_path)
    print("✅ Modelo e vetorizador carregados com sucesso!")
else:
    print("❌ ERRO: Arquivo do modelo ou vetorizador não encontrado!")
    exit()

# ✅ Dicionário para armazenar as estatísticas
sentiment_counts = {"positivo": 0, "negativo": 0, "neutro": 0}

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Texto vazio"}), 400

    text_vectorized = vetorizador.transform([text])
    probas = modelo.predict_proba(text_vectorized)[0]
    positivo_prob = probas[1]

    if positivo_prob > 0.6:
        sentimento = "positivo"
    elif positivo_prob < 0.4:
        sentimento = "negativo"
    else:
        sentimento = "neutro"

    # ✅ Atualizar estatísticas
    sentiment_counts[sentimento] += 1

    return jsonify({
        "sentiment": f"{sentimento} 😊" if sentimento == "positivo" else f"{sentimento} 😞" if sentimento == "negativo" else f"{sentimento} 😐",
        "confidence": f"{positivo_prob:.2f}"
    })

@app.route("/stats", methods=["GET"])
def show_stats():
    # ✅ Gerar gráfico
    fig, ax = plt.subplots()
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())

    ax.bar(labels, values, color=["green", "red", "gray"])
    ax.set_title("Estatísticas de Sentimentos")
    ax.set_ylabel("Quantidade")
    ax.set_xlabel("Sentimento")

    # ✅ Salvar gráfico em memória
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')

# ✅ Rodar a API
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
