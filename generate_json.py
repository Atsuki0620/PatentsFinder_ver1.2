# --------------------------------------------
# 1. 共通設定・インポート
# --------------------------------------------
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import re
import json

# --------------------------------------------
# 2. ページ設定・タイトル・説明
# --------------------------------------------
st.set_page_config(page_title="💡 PatentsFinder_ver1.2 チャット版", page_icon="🔍")
st.title("🔍 PatentsFinder_ver1.2 - チャットフロー実装 (方針3)")
st.write(
    """
    このチャット版アプリは、対話形式で IPC コードを提案し、
    その後ユーザーが「countries, assignees, publication_from」を入力すると
    自動的にJSON形式の検索パラメータを生成します。
    """
)

# --------------------------------------------
# 3. API キー入力
# --------------------------------------------
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("まずは OpenAI API Key を入力してください。", icon="🗝️")
    st.stop()

# --------------------------------------------
# 4. LangChain の LLM インスタンス生成
# --------------------------------------------
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
    temperature=0.2
)

# --------------------------------------------
# 5. セッションステート初期化
# --------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # 会話履歴
if "ipc_candidates" not in st.session_state:
    st.session_state.ipc_candidates = []  # 提案された IPC コード
if "expect_search_params" not in st.session_state:
    st.session_state.expect_search_params = False  # 検索パラメータ待ちフラグ

# --------------------------------------------
# 6. 関数定義: IPC コードを提案し、追加情報を促す質問をする
# --------------------------------------------
def generate_ipc_candidates(user_input: str):
    # 会話履歴にユーザー発言を追加
    st.session_state.messages.append({"role": "user", "content": user_input})

    # プロンプトを作成
    system_prompt = """
    あなたは「特許調査アシスタント」です。
    ユーザーが入力した技術領域やキーワードに基づき、関連性の高い IPC コードを 3～5 個程度、一言説明付きで提案してください。
    その後、次のように追加情報をリクエストしてください：
    「次に、対象国(countries)、出願人(assignees)、公開日下限(publication_from)を
    カンマ区切りで教えてください。例：JP, Sony, 2021-01-01」
    """
    lc_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    # LLM コール
    response = llm(lc_messages)
    ai_content = response.content.strip()

    # AI 応答を会話履歴に追加・表示
    st.session_state.messages.append({"role": "assistant", "content": ai_content})
    with st.chat_message("assistant"):
        st.markdown(ai_content)

    # IPC コード部分を正規表現で抽出
    codes = re.findall(r"[A-Z]\d{2}[A-Z]?\s*\d{1,2}/\d{1,2}", ai_content)
    unique_codes = []
    for code in codes:
        code_clean = code.replace(" ", "")
        if code_clean not in unique_codes:
            unique_codes.append(code_clean)
    st.session_state.ipc_candidates = unique_codes

    # 追加情報待ちフラグを立てる
    st.session_state.expect_search_params = True

# --------------------------------------------
# 7. 関数定義: 検索パラメータ用 JSON を生成する （LLM にパースを任せる版）
# --------------------------------------------
def finalize_search_parameters(user_input: str):
    """
    ユーザー入力（自由形式）を LLM に渡し、'countries', 'assignees', 'publication_from' を
    推測してもらい、さらに ipc_candidates と組み合わせて最終 JSON を生成・表示します。
    """
    # 会話履歴にユーザー発言を追加
    st.session_state.messages.append({"role": "user", "content": user_input})

    # ①：まず LLM に「パース用プロンプト」を生成
    parse_prompt = f"""
    次のユーザー入力を解析し、以下のキーを含む JSON オブジェクトを返してください。
    ・countries: 国コード (例: 日本→\"JP\", アメリカ→\"US\", 中国→\"CN\" 等)
    ・assignees: 出願人名 (そのまま文字列)
    ・publication_from: 公開日下限を \"YYYY-MM-DD\" 形式で指定 (例: \"2021年以降\"→\"2021-01-01\")
    余計な説明は一切不要で、必ず純粋に JSON オブジェクトだけを返してください。

    ユーザー入力:
    \"\"\"{user_input}\"\"\"
    """

    # LLM呼び出し（パース用）
    parse_response = llm([SystemMessage(content=parse_prompt)])
    try:
        # LLM の出力を JSON パースして dict に変換
        parsed = json.loads(parse_response.content.strip())
    except json.JSONDecodeError:
        # もし JSON 化に失敗したら、再度明確なフォーマットを要求
        follow_up = (
            "申し訳ありません。入力の形式がうまく解釈できませんでした。\n"
            "「countries, assignees, publication_from」をカンマ区切りで教えてください。"
            "例: JP, Sony, 2021-01-01"
        )
        st.session_state.messages.append({"role": "assistant", "content": follow_up})
        with st.chat_message("assistant"):
            st.markdown(follow_up)
        return

    # ②：JSON から各フィールドを取得
    countries = parsed.get("countries", [])
    assignees = parsed.get("assignees", [])
    publication_from = parsed.get("publication_from", "")

    # ③：既存の ipc_candidates と合成して最終出力を作る
    result = {
        "ipc_codes": st.session_state.ipc_candidates,
        "countries": countries,
        "assignees": assignees,
        "publication_from": publication_from
    }
    json_result = json.dumps(result, ensure_ascii=False, indent=2)

    # ④：画面表示用メッセージ
    final_message = (
        "以下が最終的な検索条件です。\n"
        f"```json\n{json_result}"
    )
    st.session_state.messages.append({"role": "assistant", "content": final_message})
    with st.chat_message("assistant"):
        st.markdown(final_message)

    # パース待ちフラグをリセット
    st.session_state.expect_search_params = False
# --------------------------------------------
# 8. 会話履歴を画面に表示
# --------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------
# 9. チャット入力フォーム
# --------------------------------------------
user_input = st.chat_input("入力してください…")

if user_input:
    if not st.session_state.ipc_candidates:
        # IPC候補がまだない → IPC候補生成ステップ
        generate_ipc_candidates(user_input)
    elif st.session_state.expect_search_params:
        # IPC候補はあるが検索パラメータ待ち → JSON生成ステップ
        finalize_search_parameters(user_input)
    else:
        # それ以外（新規のキーワードなど） → 再度候補生成から始める
        st.session_state.ipc_candidates = []
        st.session_state.expect_search_params = False
        generate_ipc_candidates(user_input)
