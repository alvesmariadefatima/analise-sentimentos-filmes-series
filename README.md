# 🎥 **Análise de Sentimentos de Comentários de Filmes e Séries**

Este projeto tem como objetivo realizar a **análise de sentimentos** em comentários de filmes e séries, utilizando técnicas de **processamento de linguagem natural (NLP)** e **machine learning** para classificar os sentimentos como **positivos**, **negativos** ou **neutros**. Ele oferece uma plataforma onde os usuários podem inserir seus próprios comentários e obter a avaliação do sentimento presente.

---

## 📜 **Sobre o Projeto**

A **Análise de Sentimentos** é uma tarefa fundamental dentro da área de **Processamento de Linguagem Natural (NLP)**. Neste projeto, desenvolvemos um modelo que classifica os sentimentos de comentários de filmes e séries, com a aplicação prática em um ambiente simples e intuitivo.

Este sistema é ideal para entender o que as pessoas estão dizendo sobre filmes e séries em suas plataformas de streaming favoritas, como Netflix, Amazon Prime e outras. A ferramenta pode ser usada por analistas de mídia, desenvolvedores e até mesmo usuários curiosos que querem entender melhor os comentários das obras que estão assistindo.

---

## 🔧 **Tecnologias Utilizadas**

Este projeto foi desenvolvido utilizando uma combinação de tecnologias poderosas, que incluem:

- **Python**: Linguagem principal do projeto.
- **Flask**: Framework web para criação de APIs e integração com a interface de usuário.
- **Scikit-learn**: Biblioteca para criação e treinamento de modelos de aprendizado de máquina.
- **Joblib**: Usado para salvar e carregar o modelo treinado.
- **Pandas**: Para manipulação e análise de dados, especialmente útil para o tratamento do conjunto de dados.
- **HTML/CSS**: Para a criação da interface de usuário simples e responsiva.
- **JavaScript**: Para interação dinâmica, como mostrar resultados de forma imediata.

---

## 🛠️ **Funcionalidades**

- **Interface Simples**: Página com um formulário onde o usuário insere seu comentário.
- **Análise de Sentimentos**: O sistema classifica automaticamente os sentimentos do comentário em **positivo**, **negativo** ou **neutro**.
- **Feedback Visual**: Após a análise, o usuário recebe um resultado visual fácil de entender sobre o sentimento do seu comentário.

---

## 🧑‍💻 **Como Rodar o Projeto**

### **1. Clone o Repositório**

Primeiro, faça o clone deste repositório para sua máquina local.

```bash
git clone https://github.com/seu-usuario/analise-sentimentos-filmes-series.git
cd analise-sentimentos
```

### **2. Instale as Dependências**

Utilize o `pip` para instalar as dependências do projeto. Crie e ative um ambiente virtual (opcional, mas recomendado).

```bash
# Criar ambiente virtual (opcional)
python -m venv venv
# Ativar ambiente virtual
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

# Instalar as dependências
pip install -r requirements.txt
```

### **3. Prepare o Modelo**

O modelo de análise de sentimentos deve ser treinado com uma base de dados antes de ser usado. Caso já tenha o modelo treinado (arquivo `.pkl`), você pode pular essa etapa. Caso contrário, execute o script para treinar o modelo:

```bash
python train_model.py
```

### **4. Rodar a API**

Para rodar a aplicação Flask localmente, execute o comando abaixo:

```bash
python app.py
```

A API estará disponível em `http://127.0.0.1:5000/`.

### **5. Acesse o Frontend**

Abra o arquivo `index.html` em um navegador ou crie um servidor para servir a página HTML (se estiver usando Flask, o frontend já será servido pela API). Em seguida, insira um comentário no campo de texto e envie para obter a análise de sentimento.

---

## 📊 **Exemplo de Uso**

Após rodar o projeto, o usuário poderá inserir um comentário como o exemplo abaixo:

**Comentário**:  
_"O filme foi incrível! Adorei a história e os personagens."_

**Resultado Esperado**:  
Sentimento: **Positivo**

---

## 🔄 **Como Contribuir**

Se você deseja contribuir para o projeto, siga as etapas abaixo:

1. **Fork** o repositório.
2. Crie uma nova branch (`git checkout -b minha-nova-funcionalidade`).
3. Faça suas alterações e commite (`git commit -am 'Adicionando nova funcionalidade'`).
4. Faça o **push** para sua branch (`git push origin minha-nova-funcionalidade`).
5. Envie um **pull request**.

---

## 📜 **Licença**

Este projeto está licenciado sob a **MIT License**. Veja o arquivo `LICENSE` para mais detalhes.

---

