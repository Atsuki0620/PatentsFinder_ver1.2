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
# 7. 関数定義: 検索パラメータ用 JSON を生成する
# --------------------------------------------
def finalize_search_parameters(user_input: str):
    # 会話履歴にユーザー発言を追加
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 入力を countries, assignees, publication_from に分割
    parts = [part.strip() for part in user_input.split(",")]
    if len(parts) < 3:
        # まだ不十分なので再度リクエスト
        follow_up = "入力が不十分です。countries, assignees, publication_from をカンマ区切りで正確に入力してください。例：JP, Sony, 2021-01-01"
        st.session_state.messages.append({"role": "assistant", "content": follow_up})
        with st.chat_message("assistant"):
            st.markdown(follow_up)
        return

    countries = [c.strip() for c in parts[0].split()]
    assignees = [a.strip() for a in parts[1].split()]
    publication_from = parts[2]

    result = {
        "ipc_codes": st.session_state.ipc_candidates,
        "countries": countries,
        "assignees": assignees,
        "publication_from": publication_from
    }
    json_result = json.dumps(result, ensure_ascii=False, indent=2)

    final_message = f"以下が最終的な検索条件です。\n```json\n{json_result}\n```\n以上が特許検索条件になります。このまま検索エンジンに貼り付けて調査を開始してください。"
    st.session_state.messages.append({"role": "assistant", "content": final_message})
    with st.chat_message("assistant"):
        st.markdown(final_message)

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
