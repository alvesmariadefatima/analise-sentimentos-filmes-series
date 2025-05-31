from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import os
import pandas as pd
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
        "sentiment": f"{sentimento} 😊" if sentimento == "positivo"
                    else f"{sentimento} 😞" if sentimento == "negativo"
                    else f"{sentimento} 😐",
        "confidence": f"{positivo_prob:.2f}"
    })


@app.route("/comentarios")
def comentarios():
    df = pd.read_csv('dataset/IMDB_Dataset.csv', encoding='utf-8', on_bad_lines='skip')

    def safe_sample(df, sentiment_label, n=3):
        df_filtered = df[df['sentiment'] == sentiment_label]
        if len(df_filtered) == 0:
            return pd.DataFrame()
        return df_filtered.sample(n=min(n, len(df_filtered)))

    positivos = safe_sample(df, 'positive', 3)
    negativos = safe_sample(df, 'negative', 3)

    todos = pd.concat([positivos, negativos]).sample(frac=1).reset_index(drop=True)
    return jsonify(todos.to_dict(orient='records'))


@app.route("/stats", methods=["GET"])
def show_stats():
    # ✅ Gerar gráfico com fundo escuro e cores compatíveis
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())

    background_color = '#1B2430'
    text_color = '#ffffff'
    bar_colors = ['#f57524', '#3081bf', '#ffffff']

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=background_color)
    ax.set_facecolor(background_color)

    bars = ax.bar(labels, values, color=bar_colors)

    ax.set_title("Estatísticas de Sentimentos", color=text_color)
    ax.set_ylabel("Quantidade", color=text_color)
    ax.set_xlabel("Sentimento", color=text_color)

    ax.tick_params(colors=text_color)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    color=text_color)

    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor=background_color, bbox_inches='tight', transparent=True)
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')


# ✅ Rodar a API
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
