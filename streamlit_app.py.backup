import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

# ------------------------------------------------------------
# 0. SessionState の初期化
# ------------------------------------------------------------
if "phase" not in st.session_state:
    st.session_state["phase"] = "api_key_input"

if "openai_key" not in st.session_state:
    st.session_state["openai_key"] = ""

if "user_inputs" not in st.session_state:
    st.session_state["user_inputs"] = []

if "latest_plan" not in st.session_state:
    st.session_state["latest_plan"] = ""

if "json_params" not in st.session_state:
    st.session_state["json_params"] = ""

if "greeted" not in st.session_state:
    st.session_state["greeted"] = False

if "plan_shown" not in st.session_state:
    st.session_state["plan_shown"] = False

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# ------------------------------------------------------------
# 1. Chat 履歴を画面に描画する関数
# ------------------------------------------------------------
def render_chat_history():
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ------------------------------------------------------------
# 2. フェーズ①：APIキー入力用 UI（修正版）
# ------------------------------------------------------------
def phase_api_key_input():
    st.title("💬 PatentsFinder_ver1.2")
    st.write(
        "このアプリは特許調査をサポートするチャット形式のツールです。\n"
        "最初に OpenAI API キーを入力し、「送信」ボタンを押してください。"
    )

    # すでにセッションに API キーがあれば次フェーズへ
    if st.session_state["openai_key"]:
        st.session_state["phase"] = "awaiting_query"
        return

    # まだキーが空のときだけ、テキスト入力と送信ボタンを表示
    key_input = st.text_input(
        "OpenAI API Key", 
        type="password", 
        value=st.session_state["openai_key"]
    )

    # 「送信」ボタンを押したらフェーズ遷移
    if st.button("送信"):
        if key_input:
            st.session_state["openai_key"] = key_input
            st.session_state["phase"] = "awaiting_query"
            st.session_state["greeted"] = False
            return
        else:
            st.error("API キーが入力されていません。再度ご確認ください。")


# ------------------------------------------------------------
# 3. フェーズ②：ユーザー調査ニーズヒアリング
# ------------------------------------------------------------
def phase_awaiting_query():
    if not st.session_state["greeted"]:
        greeting = (
            "こんにちは！どんな特許情報をお探しですか？\n"
            "気になる技術や公開日、調査したい国やキーワードなど、思いつくまま教えてください！"
        )
        st.session_state["chat_history"].append({"role": "assistant", "content": greeting})
        st.session_state["greeted"] = True

    render_chat_history()

    if user_query := st.chat_input("調査ニーズを入力してください…"):
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        st.session_state["user_inputs"].append(user_query)
        st.session_state["phase"] = "proposed_plan"


# ------------------------------------------------------------
# 4. フェーズ③：検索方針提案（自然言語）
# ------------------------------------------------------------
def phase_proposed_plan():
    if not st.session_state["plan_shown"]:
        render_chat_history()

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=st.session_state["openai_key"],
            temperature=0.7,
            streaming=False,
        )

        prompt_lines = []
        for idx, txt in enumerate(st.session_state["user_inputs"], start=1):
            prompt_lines.append(f"{idx}. {txt}")
        joined_input = "\n".join(prompt_lines)

        system_prompt = (
            "あなたは特許調査支援アシスタントです。"
            "以下のユーザーの要望をもとに「技術分野」「公開日」「出願国」「キーワード」"
            "を含む検索方針を自然言語で1～2文程度で提案してください。"
        )
        human_prompt = (
            "ユーザーからの調査ニーズ:\n"
            f"{joined_input}\n\n"
            "上記を踏まえて、簡潔に検索方針を示してください。"
        )

        messages = [
            HumanMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = llm(messages)
        plan_text = response.content.strip()

        st.session_state["latest_plan"] = plan_text
        st.session_state["chat_history"].append({"role": "assistant", "content": plan_text})
        st.session_state["plan_shown"] = True

        render_chat_history()
    else:
        render_chat_history()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("この方針で検索実行"):
            st.session_state["phase"] = "generate_json"
    with col2:
        if st.button("方針を修正する"):
            st.session_state["plan_shown"] = False
            st.session_state["phase"] = "awaiting_query"


# ------------------------------------------------------------
# 5. フェーズ④：JSON形式の検索パラメータ生成
# ------------------------------------------------------------
def phase_generate_json():
    render_chat_history()

    if not st.session_state["json_params"]:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=st.session_state["openai_key"],
            temperature=0.0,
            streaming=False,
        )

        natural_plan = st.session_state["latest_plan"]
        system_prompt = (
            "あなたは特許データベース検索支援アシスタントです。\n"
            "以下の自然言語の検索方針を厳密に JSON フォーマットで出力してください。"
        )
        human_prompt = (
            f"自然言語の検索方針:\n「{natural_plan}」\n\n"
            "この方針をもとに、以下の形式の JSON オブジェクトだけを返してください。\n"
            "```\n"
            "{\n"
            '  "technical_field": "<技術分野>",\n'
            '  "public_date_from": "YYYY-MM-DD",\n'
            '  "public_date_to": "YYYY-MM-DD",\n'
            '  "countries": ["<国コード1>", "<国コード2>", ...],\n'
            '  "keywords": ["<キーワード1>", "<キーワード2>", ...]\n'
            "}\n"
            "```\n"
            "※ フィールド名は厳密に上記の名前を使い、他のキーを追加しないでください。\n"
            "※ 出力は JSON オブジェクトだけとし、余計な説明や文章を含めないでください。"
        )

        messages = [
            HumanMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = llm(messages)
        json_text = response.content.strip()

        st.session_state["json_params"] = json_text
        st.session_state["chat_history"].append({"role": "assistant", "content": "検索パラメータ（JSON形式）を生成しました。"})
        render_chat_history()

    st.markdown("**▼ 以下が生成された検索パラメータ（JSON形式）です ▼**")
    st.code(st.session_state["json_params"], language="json")


# ------------------------------------------------------------
# 6. main 関数相当：fase によって処理を振り分け
# ------------------------------------------------------------
def main():
    if st.session_state["phase"] == "api_key_input":
        phase_api_key_input()
    elif st.session_state["phase"] == "awaiting_query":
        phase_awaiting_query()
    elif st.session_state["phase"] == "proposed_plan":
        phase_proposed_plan()
    elif st.session_state["phase"] == "generate_json":
        phase_generate_json()
    else:
        st.session_state["phase"] = "api_key_input"
        phase_api_key_input()


if __name__ == "__main__":
    main()
