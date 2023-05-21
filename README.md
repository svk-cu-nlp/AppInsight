# AppInsight: Extracting Software Refactoring Insights from App Reviews
<h2 align="center"> AppInsight Tool Demo </h2>
<video src="https://github.com/svk-cu-nlp/AppInsight/assets/96056131/b00e1ea7-e494-43c6-8693-2ea195730065"></video>


## 🔗 Features
- Sentiment analysis of reviews
- Topic analysis of reviews
- Summary generation of each review
- Querying agent to get clearer idea about which feature requires rework

## Getting Started
### ⚙️ Setup
```bash
pip install -r requirements.txt
```
### 🔌 Setting OpenAI API
- Get an OpenAI [API Key](https://platform.openai.com/account/api-keys)
- Add the API Key and Organization Key in [config.py][src/config.py]
### 💻 Execution
```bash
streamlit run ./src/app.py
```

