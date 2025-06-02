# --------------------------------------------
# 1. 共通設定・インポート
# --------------------------------------------
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# --------------------------------------------
# 2. ページ設定・タイトル・説明
# --------------------------------------------
st.set_page_config(page_title="💡 PatentsFinder ver1.2", page_icon="🔍")
st.title("🔍 PatentsFinder ver1.2")
st.write(
    """
    このアプリは、会話を通じて「特許調査に必要な IPC (International Patent Classification) コード」を
    絞り込むための AI アシスタントです。

    1. 左のサイドバー（または上部）の入力欄に OpenAI API Key を入力  
    2. 下部のテキストフィールドで「調査したい技術領域やキーワード」を入力し、  
       ・「関連技術提案」を押すと技術サブトピックを提案  
       ・「IPCコード生成」を押すと IPC コード案を生成  
    3. 応答は会話UIに流れます。  
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
# 5. session_state で会話履歴を管理
# --------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user"/"assistant", "content": "..."} のリスト

# --------------------------------------------
# 6. (関数定義) 関連技術を提案する関数
# --------------------------------------------
def suggest_technologies(user_input: str):
    """
    ユーザー入力に対し、関連性の高い技術分野やサブトピックを自然文で 2～4 項目程度提案する。
    IPCコードには一切言及しない。
    """
    # 会話履歴にユーザー発言を追加
    st.session_state.messages.append({"role": "user", "content": user_input})

    # SystemMessage: 関連技術提案専用プロンプト
    system_prompt = """
    あなたは「特許調査アシスタント」です。以下のルールに従って、ユーザーが入力した技術領域やキーワードに対し、
    関連性の高い技術分野やサブトピックを 2〜4 項目、自然な日本語で提案してください。IPCコードや特許分類番号には言及せず、
    クリーンに技術的視点のみで回答します。

    例：ユーザーが「逆浸透膜の機械学習」と入力した場合、
     1. 逆浸透膜プロセスにおける運転条件最適化のための機械学習アルゴリズム開発
     2. センサー収集データを用いた膜汚染検知および予測技術
     3. AIを活用した膜製造工程での材料選定と品質管理
    """
    # LangChain 用メッセージリストを生成
    lc_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    # LLM コール
    response = llm(lc_messages)
    ai_content = response.content.strip()

    # AI 応答をセッションに追加
    st.session_state.messages.append({"role": "assistant", "content": ai_content})

    # 画面に表示
    with st.chat_message("assistant"):
        st.markdown(ai_content)


# --------------------------------------------
# 7. (関数定義) IPCコードを生成する関数
# --------------------------------------------
def generate_ipc_codes(user_input: str):
    """
    ユーザー入力に対し、関連する IPC コードを 3〜5 個程度提案し、
    各コードの技術範囲を一言で説明する。
    """
    # 会話履歴にユーザー発言を追加
    st.session_state.messages.append({"role": "user", "content": user_input})

    # SystemMessage: IPCコード生成専用プロンプト
    system_prompt = """
    あなたは「特許調査アシスタント」です。ユーザーが入力した技術領域やキーワードに基づき、関連性の高い IPC コードを 3～5 個程度、一言説明付きで提案してください。
    たとえば、ユーザーが「逆浸透膜の機械学習」と入力した場合、下記のように、IPCコードと説明を Markdown の箇条書きで出力してください。
     - C02F 1/00：逆浸透膜の製造や利用
     - G06N 3/04：機械学習に関連する知識処理
     - G01N 33/569：逆浸透膜の性能評価
     
    以下の**厳密な JSON**形式の検索パラメータを生成してください。
    絶対に余計な説明文を含まず、**純粋な JSON オブジェクト**だけを返してください：
    {
      "ipc_codes": ["B01D61/02", "B01D61/08", "C02F1/44"],
      "assignees": [],
      "publication_from": "YYYY-MM-DD"
    }
    """
    # LangChain 用メッセージリストを生成
    lc_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    # LLM コール
    response = llm(lc_messages)
    ai_content = response.content.strip()

    # AI 応答をセッションに追加
    st.session_state.messages.append({"role": "assistant", "content": ai_content})

    # 画面に表示
    with st.chat_message("assistant"):
        st.markdown(ai_content)


# --------------------------------------------
# 8. 会話履歴を画面に表示
# --------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------
# 9. ユーザー入力フォームとボタン配置
# --------------------------------------------
user_input = st.text_input("調査したい技術領域やキーワードを入力してください…")

if st.button("関連技術提案"):
    if user_input:
        generate_ipc = False
        suggest_technologies(user_input)
    else:
        st.warning("まずは入力欄に技術領域やキーワードを入力してください。")

if st.button("IPCコード生成"):
    if user_input:
        generate_ipc_codes(user_input)
    else:
        st.warning("まずは入力欄に技術領域やキーワードを入力してください。")

# --------------------------------------------
# 10. フッターや追加情報（必要に応じて）
# --------------------------------------------
st.write("---")
st.write("※「関連技術提案」と「IPCコード生成」はそれぞれ別の処理を行います。")


おまけのテスト
# サイドバーの使用例
option = st.sidebar.selectbox('オプションを選択', ['オプション1', 'オプション2', 'オプション3'])
st.write(f'選択したオプションは {option} です。')

# カラムの使用例
column1, column2, column3 = st.columns(3)
with column1:
    st.write('これはカラム1です。')
with column2:
    st.write('これはカラム2です。')
with column3:
    st.write('これはカラム3です。')

col1, col2 = st.columns([2, 3])
col1.metric(label="メトリック1", value=123)
col2.metric(label="メトリック2", value=456)

col1、col2 = st.columns([2、3])
with col1:
    st.metric(label="メトリック1"、value=123)
    st.caption("これはメトリック1に関する追加の情報です。")
with col2:
    st.metric(label="メトリック2"、value=456)
    st.caption("これはメトリック2に関する追加の情報です。")

with st.expander("クリックして展開"):
    st.write("非表示のコンテンツ")
