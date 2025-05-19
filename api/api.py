from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# ✅ Carregar o modelo e o vetorizador
modelo_path = "modelo_sentimentos.pkl"
vetorizador_path = "vetorizador.pkl"

if os.path.exists(modelo_path) and os.path.exists(vetorizador_path):
    modelo = joblib.load(modelo_path)
    vetorizador = joblib.load(vetorizador_path)
    print("✅ Modelo e vetorizador carregados com sucesso!")
else:
    print("❌ ERRO: Arquivo do modelo ou vetorizador não encontrado!")
    exit()

# ✅ Criar rota da API para análise de sentimentos
@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Texto vazio"}), 400

    # ✅ Transformar texto para o formato do modelo
    text_vectorized = vetorizador.transform([text])
    
    # ✅ Obter a probabilidade da previsão
    probas = modelo.predict_proba(text_vectorized)[0]  # Probabilidades para [negativo, positivo]
    positivo_prob = probas[1]

    # ✅ Definir sentimento com base na probabilidade
    if positivo_prob > 0.6:
        sentimento = "positivo 😊"
    elif positivo_prob < 0.4:
        sentimento = "negativo 😞"
    else:
        sentimento = "neutro 😐"

    return jsonify({"sentiment": sentimento, "confidence": f"{positivo_prob:.2f}"})

# ✅ Rodar a API
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
