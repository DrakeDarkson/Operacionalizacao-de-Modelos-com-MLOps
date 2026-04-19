# Operacionalização de Modelos com MLOps

Projeto desenvolvido na disciplina de MLOps com foco na estruturação, experimentação e operacionalização de um sistema de machine learning completo.

## 📌 Objetivo

Desenvolver um sistema de classificação capaz de identificar municípios com alta vulnerabilidade socioeconômica, estruturado de forma reprodutível, rastreável e operacionalizável.

---

## 🧠 Problema

Classificação binária de municípios brasileiros em:

- Alta vulnerabilidade
- Baixa vulnerabilidade

Baseado em indicadores socioeconômicos (2017).

---

## 📊 Dataset

Fonte: integração de dados públicos (SNIS, DATASUS, Censo Escolar, RAIS)

Arquivo utilizado:
data/raw/censo_municipal.csv

Target:
% de extremamente pobres no Cadastro Único pós Bolsa Família 2017

---

## ⚙️ Tecnologias

- Python 3.11
- scikit-learn
- MLflow
- Flask
- Pandas / NumPy
- GitHub Actions

---

## 🚀 Execução do Projeto

### 1. Instalar dependências

pip install -r requirements.txt

### 2. Executar experimentos

python main.py

### 3. Visualizar experimentos (MLflow)

mlflow ui --backend-store-uri sqlite:///mlflow.db

Acesse:
http://127.0.0.1:5000

### 4. Empacotar modelo final

python -m scripts.package_final_model

### 5. Testar inferência

python -m scripts.smoke_inference

### 6. Subir serviço de inferência

python -m src.serving.app

Endpoints:

- GET /health
- POST /predict

### 7. Simular monitoramento pós-deploy

python -m scripts.simulate_post_deploy

---

## 🤖 Modelos Avaliados

- Perceptron (baseline)
- Decision Tree (com e sem tuning)
- Random Forest (modelo final)

Melhor modelo:
Random Forest

---

## 📈 Métricas

- Accuracy
- Precision
- Recall
- F1-score

Métrica principal:
F1-score

---

## 💼 Métricas de Negócio

- Recall da classe positiva
- Taxa de falso negativo
- Municípios corretamente priorizados

---

## 🧪 Experimentação

- Pipelines end-to-end
- Validação cruzada
- Grid Search
- Registro no MLflow

---

## 📉 Redução de Dimensionalidade

- PCA
- LDA

Resultado:
Sem ganho significativo → modelo final sem redução

---

## 📦 Deploy

- Modelo persistido com joblib
- Versionamento
- Pacote de inferência

---

## 🌐 API de Inferência

Flask:
- Carregamento automático do modelo
- Predição via JSON
- Endpoint de saúde

---

## 🔄 Monitoramento

- Métricas pós-deploy
- Métricas de negócio
- Drift de dados (KS Test)
- Drift de modelo

---

## 📊 MLflow

- Experimentos
- Deploy
- Monitoramento
- Artifacts

---

## 🔁 CI/CD

GitHub Actions:
- Instala dependências
- Empacota modelo
- Testa inferência

---

## ⚠️ Limitações

- Dados de 2017
- Possível viés
- Dados ausentes
- Generalização limitada

---

## 👨‍💻 Autor

Breno dos Santos Mota
