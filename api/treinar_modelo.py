import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# ✅ Carregar o dataset
try:
    df = pd.read_csv("dataset/IMDB_Dataset.csv")
    print("✅ Dataset carregado com sucesso!")
except FileNotFoundError:
    print("❌ ERRO: O arquivo 'IMDB_Dataset.csv' não foi encontrado.")
    exit()

# ✅ Preparação dos dados
df = df.sample(frac=1).reset_index(drop=True)  # Embaralhar os dados
X = df["review"]  # Texto das avaliações
y = df["sentiment"].map({"positive": 1, "negative": 0})  # Converter para 1 (positivo) e 0 (negativo)

# ✅ Vetorização do texto
vetorizador = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = vetorizador.fit_transform(X)

# ✅ Divisão dos dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# ✅ Treinar o modelo
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# ✅ Salvar o modelo e o vetorizador
joblib.dump(modelo, "modelo_sentimentos.pkl")
joblib.dump(vetorizador, "vetorizador.pkl")

print("🎯 Modelo treinado e salvo com sucesso!")