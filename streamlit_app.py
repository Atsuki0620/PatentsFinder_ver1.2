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
    # ユーザーが入力したヒアリングテキストを格納するリスト
    st.session_state["user_inputs"] = []

if "latest_plan" not in st.session_state:
    # LLM から得た「自然言語での検索方針」文字列
    st.session_state["latest_plan"] = ""

if "json_params" not in st.session_state:
    # LLM から得た「JSON形式の検索パラメータ」文字列
    st.session_state["json_params"] = ""

if "greeted" not in st.session_state:
    # 初回ヒアリング時の挨拶を一度だけ出すフラグ
    st.session_state["greeted"] = False

if "plan_shown" not in st.session_state:
    # 方針提案を一度だけ実行・表示したかどうかのフラグ
    st.session_state["plan_shown"] = False

if "chat_history" not in st.session_state:
    # 画面に表示するチャット履歴（一連のメッセージ）を格納するリスト
    # 各要素は {"role": "assistant" or "user", "content": "..."} の辞書
    st.session_state["chat_history"] = []


# ------------------------------------------------------------
# 1. Chat 履歴を画面に描画する関数
# ------------------------------------------------------------
def render_chat_history():
    """
    st.session_state["chat_history"] に蓄積されたメッセージを
    for ループで st.chat_message() を使い描画する。
    """
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ------------------------------------------------------------
# 2. フェーズ①：APIキー入力用 UI
# ------------------------------------------------------------
def phase_api_key_input():
    st.title("💬 PatentsFinder_ver1.2")
    st.write(
        "このアプリは特許調査をサポートするチャット形式のツールです。\n"
        "最初に OpenAI API キーを入力してください。"
    )

    # API キーを入力させるテキストボックス
    key_input = st.text_input("OpenAI API Key", type="password")

    if key_input:
        # 入力されたらセッションに保存して次フェーズへ移行
        st.session_state["openai_key"] = key_input
        st.session_state["phase"] = "awaiting_query"
        # 挨拶フラグをリセット（もし再度戻ってきても再表示させたいなら）
        st.session_state["greeted"] = False

    # このフェーズではチャット履歴表示はしないため return で終了
    return


# ------------------------------------------------------------
# 3. フェーズ②：ユーザー調査ニーズヒアリング
# ------------------------------------------------------------
def phase_awaiting_query():
    # 挨拶メッセージを一度だけ表示
    if not st.session_state["greeted"]:
        greeting = (
            "こんにちは！どんな特許情報をお探しですか？\n"
            "気になる技術や公開日、調査したい国やキーワードなど、思いつくまま教えてください！"
        )
        # チャット履歴に追加して画面描画
        st.session_state["chat_history"].append({"role": "assistant", "content": greeting})
        st.session_state["greeted"] = True

    # 画面に既存チャット履歴を描画
    render_chat_history()

    # ユーザーから自由入力を受け付ける
    if user_query := st.chat_input("調査ニーズを入力してください…"):
        # ユーザーの入力を履歴に追加
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        # ヒアリングテキストのリストにも保存
        st.session_state["user_inputs"].append(user_query)
        # 次フェーズへ移行
        st.session_state["phase"] = "proposed_plan"


# ------------------------------------------------------------
# 4. フェーズ③：検索方針提案（自然言語）
# ------------------------------------------------------------
def phase_proposed_plan():
    # まだ方針を生成していなければ LLM をコールして自然言語方針を取得
    if not st.session_state["plan_shown"]:
        # ① 画面に既存チャット履歴を描画
        render_chat_history()

        # ② LangChain LLM インスタンスを作成
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=st.session_state["openai_key"],
            temperature=0.7,
            streaming=False,
        )

        # ③ ヒアリング情報をまとめたプロンプトを作成
        prompt_lines = []
        for idx, txt in enumerate(st.session_state["user_inputs"], start=1):
            prompt_lines.append(f"{idx}. {txt}")
        joined_input = "\n".join(prompt_lines)

        # ④ LLM に渡すメッセージのリストを作成
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

        # ⑤ LLM をコールして返答を取得
        response = llm(messages)
        plan_text = response.content.strip()

        # ⑥ 取得結果をセッションに保存し、チャット履歴にも追加
        st.session_state["latest_plan"] = plan_text
        st.session_state["chat_history"].append({"role": "assistant", "content": plan_text})
        st.session_state["plan_shown"] = True

        # ⑦ 画面にチャット履歴を描画
        render_chat_history()

    else:
        # すでに plan_shown=True なら、チャット履歴を描画するだけ
        render_chat_history()

    # ⑧ 方針が表示された直後に「検索実行」「方針を修正」ボタンを並べて表示
    col1, col2 = st.columns(2)
    with col1:
        if st.button("この方針で検索実行"):
            st.session_state["phase"] = "generate_json"
    with col2:
        if st.button("方針を修正する"):
            # フラグをリセットして再度フェーズ②へ
            st.session_state["plan_shown"] = False
            # ヒアリングをやり直すためフェーズを戻す
            st.session_state["phase"] = "awaiting_query"


# ------------------------------------------------------------
# 5. フェーズ④：JSON形式の検索パラメータ生成
# ------------------------------------------------------------
def phase_generate_json():
    # まずはチャット履歴を描画
    render_chat_history()

    # まだ JSON を生成していなければ LLM をコールして JSON を取得
    if not st.session_state["json_params"]:
        # LangChain LLM インスタンスを作成
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=st.session_state["openai_key"],
            temperature=0.0,  # JSON 生成なので deterministic にしておく
            streaming=False,
        )

        # 自然言語方針を埋め込むプロンプトを作成
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

        # LLM にリクエストして返答を取得
        response = llm(messages)
        json_text = response.content.strip()

        # セッションに保存し、チャット履歴にも一度だけ追加
        st.session_state["json_params"] = json_text
        st.session_state["chat_history"].append({"role": "assistant", "content": "検索パラメータ（JSON形式）を生成しました。"})
        # JSON 本体はチャット履歴ではなく、コードブロックで画面に表示する

        # 画面にチャット履歴を再描画
        render_chat_history()

    # 生成された JSON をコードブロックで表示
    st.markdown("**▼ 以下が生成された検索パラメータ（JSON形式）です ▼**")
    st.code(st.session_state["json_params"], language="json")

    # （以降、SQL 記述フェーズに引き渡す際は st.session_state["json_params"] を利用します）


# ------------------------------------------------------------
# 6. main 関数相当：fase によって処理を振り分け
# ------------------------------------------------------------
def main():
    # フェーズに応じて処理を呼び出す
    if st.session_state["phase"] == "api_key_input":
        phase_api_key_input()
    elif st.session_state["phase"] == "awaiting_query":
        phase_awaiting_query()
    elif st.session_state["phase"] == "proposed_plan":
        phase_proposed_plan()
    elif st.session_state["phase"] == "generate_json":
        phase_generate_json()
    else:
        # 予期しない phase が入っていた場合は初期化フェーズに戻す
        st.session_state["phase"] = "api_key_input"
        phase_api_key_input()


if __name__ == "__main__":
    main()
