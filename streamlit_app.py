import streamlit as st

# LangChain 関連インポート
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

# ------------------------------------------------------------
# 1. タイトル・説明表示
# ------------------------------------------------------------
st.set_page_config(page_title="💬 LangChain Chatbot", page_icon="🤖")
st.title("💬 Chatbot (LangChain + Streamlit)")
st.write(
    "LangChain を使って GPT-3.5 に問い合わせるシンプルなチャットアプリです。\n"
    "最初に OpenAI API キーを入力しないとチャットを開始できません。"
)

# ------------------------------------------------------------
# 2. ユーザーに API キーを入力してもらう
# ------------------------------------------------------------
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    # API キー未入力時は注意メッセージのみ表示して終了
    st.info("API キーを入力するとチャットを開始できます。", icon="🗝️")
    st.stop()  # 以下のコードは実行されないようにする

# ------------------------------------------------------------
# 3. LangChain の LLM インスタンスを生成
# ------------------------------------------------------------
# ChatOpenAI のパラメータ例
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
    temperature=0.7,     # 必要に応じて調整
    streaming=False      # 今回はストリーミングせず、一括で結果を取得する例
)

# ------------------------------------------------------------
# 4. session_state で会話履歴を保持
# ------------------------------------------------------------
if "messages" not in st.session_state:
    # {"role": "user" or "assistant", "content": "..."} のリスト
    st.session_state.messages = []

# ------------------------------------------------------------
# 5. 既存のチャット履歴を表示
# ------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------
# 6. ユーザーからの入力を受け付ける
# ------------------------------------------------------------
if prompt := st.chat_input("質問を入力してください…"):  # 何も入れなければ何もしない
    # 6-1. ユーザーの発言を session_state に追加・表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 6-2. LangChain 用にメッセージを変換 (HumanMessage / AIMessage)
    lc_messages = []
    for message in st.session_state.messages:
        if message["role"] == "user":
            lc_messages.append(HumanMessage(content=message["content"]))
        else:
            lc_messages.append(AIMessage(content=message["content"]))

    # 6-3. LLM に問い合わせて返答を取得
    response_ai = llm(lc_messages)  # AIMessage が返る
    ai_content = response_ai.content  # 実際の文字列

    # 6-4. AI の発言を表示・履歴に追加
    with st.chat_message("assistant"):
        st.markdown(ai_content)
    st.session_state.messages.append({"role": "assistant", "content": ai_content})
