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
st.write("""
このアプリは、会話を通じて「特許調査に必要な IPC (International Patent Classification) コード」を絞り込むための AI アシスタントです。
1. 左のサイドバー（または下部）の入力欄に OpenAI API Key を入力  
2. 『調査したい技術領域やキーワード』を会話形式で入力すると、AI が関連する IPC コードを提案します。  
3. 提案された IPC コードを使って、実際の特許検索エンジンで調査できます。
""")

# --------------------------------------------
# 3. サイドバー or 最上部：API キー入力
# --------------------------------------------
# （※ 画面上部に直接置くか、st.sidebar を使うかは好みで変更可）
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("まずは OpenAI API Key を入力してください。", icon="🗝️")
    st.stop()

# --------------------------------------------
# 4. LangChain の LLM インスタンス生成
#    ── “AI の役割” を定義する system_prompt を含める
# --------------------------------------------
SYSTEM_PROMPT = """
あなたは「特許調査クエリジェネレータ」です。ユーザーとの対話を通じて、以下の検索条件を最終的に出力してください。ユーザーは自由形式で一度にすべての条件を入力できます。もしユーザーの入力にすべての条件が含まれていれば、最終的な検索条件を***JSON形式***で出力してください。不足があれば質問を行い、最終的に以下の項目を揃えてください。
• ipc_codes: リスト形式で複数のIPCコードを指定（例：["B01D 53/14", "C02F 1/00"]）
• publication_from: YYYY-MM-DD形式で公開日下限を指定（例："2020-01-01"）
• countries: 対象国コードのリスト（例： ["JP", "US", "EP"]）
• keywords: 検索に含めたいキーワードのリスト（例： ["膜ろ過", "ファウリング", "検知"] ）
──ルール──
ユーザーが自由形式で一度に入力した場合、入力文の中からpublication_from、countries、keywordsを解析し、ipc_codesはキーワードに応じて関連性の高いIPCコードを3〜5個程度、各コードの技術範囲を一言で説明しながら自動的に生成してください。ユーザーがIPCコードを指定した場合は、そのままリストに含めてください。
ユーザーの入力にすべての以下の情報が含まれていれば、追加質問なしで最終的な検索条件をMarkdownのJSONコードブロックで出力してください。
形式は以下の通り：

{
  "ipc_codes": ["選択したコード1", "選択したコード2", …],
  "publication_from": "YYYY-MM-DD",
  "countries": ["JP", "US", …],
  "keywords": ["キーワード1", "キーワード2", …]
}

ユーザーの入力が不十分で、publication_from、countries、keywordsのいずれかが欠けている場合は、不足している項目を指摘し、具体例を示しながら質問をしてください。
例："公開日下限（publication_from）が指定されていません。特定の公開年以降の特許を調査したい場合は、YYYY-MM-DD形式で教えてください（例：2020-01-01）。"
例："対象国（countries）が指定されていません。調査対象の国コードを教えてください（例：JP, US, CN）。"
例："キーワード（keywords）が指定されていません。検索に含めたいキーワードをリスト形式で教えてください（例：膜ろ過, ファウリング）。"
ユーザーが指定したIPCコードがあれば、そのままipc_codesリストに含めて良いです。指定がない場合は、keywordsを元に推測した3〜5個のIPCコードを提案してください。各コードの技術範囲を一言で説明し、ユーザーが選択しやすいように番号を振って示します。
ユーザーとの対話が完了し、すべての項目が揃ったら、最終的な検索条件をJSON形式で出力し、最後に "以上が特許検索条件になります。このまま検索エンジンに貼り付けて調査を開始してください。" という一文を付け加えてください。
──応答例──
ユーザー入力：
"逆浸透膜の機械学習についての事例 2021年以降の出願 日本、アメリカ、中国を対象"
AIが解析して即時応答できる場合：

{
  "ipc_codes": ["C02F 1/00", "G06N 3/04", "G01N 33/569"],
  "publication_from": "2021-01-01",
  "countries": ["JP", "US", "CN"],
  "keywords": ["逆浸透膜", "機械学習", "ファウリング"]
}

以上が特許検索条件になります。このまま検索エンジンに貼り付けて調査を開始してください。
──注意──
publication_from、countries、keywordsのいずれかが入力文に見当たらない場合は、必ず具体例を添えて質問してください。
IPCコードについては、ユーザーが指定した場合はそのまま利用し、指定がなければキーワードを元に提案してください。
応答は常にMarkdownのコードブロックか質問文で行い、JSON以外の余計なテキストは含めないようにしてください。
"""



llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    openai_api_key=openai_api_key,
    temperature=0.2  # IPC 絞り込みではあまりブレを持たせないためやや低め
)

# --------------------------------------------
# 5. session_state で会話履歴を管理
# --------------------------------------------
# {"role":"user"/"assistant"/"system", "content": "..."} のリストを保持
if "messages" not in st.session_state:
    # 初回起動時にシステムプロンプトを挿入しておく
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# --------------------------------------------
# 6. 既存の会話履歴を表示（system は表示せず user/assistant のみ描画）
# --------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# --------------------------------------------
# 7. ユーザー入力フォーム
# --------------------------------------------
if prompt := st.chat_input("調査したい技術領域・対象期間・出願国・キーワードを入力してください…"):

    # 7-1. ユーザー発言をセッションに追加・表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

     # 7-2. LangChain 用にメッセージを HumanMessage/AIMessage/SystemMessage に変換
    lc_messages = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
        elif m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))

    # 7-3. LLM に問い合わせ（IPC コードを絞り込む役割のまま応答）
    # ※ LangChain v0.1 系では system メッセージを自動検出しない場合があるため、
    #    その際は ChatPromptTemplate を使って system と user を分離する必要があります。
    #    ここでは単純化して「lc_messages に system/user を混在させる形」を想定。
    response = llm(lc_messages)
    ai_content = response.content

    # 7-4. AI の応答を表示・履歴に追加
    with st.chat_message("assistant"):
        st.markdown(ai_content)
    st.session_state.messages.append({"role": "assistant", "content": ai_content})

# --------------------------------------------
# 8. （オプション）提案 IPC コードのみを抽出してサマリ表示
# --------------------------------------------
# AI の回答は Markdown 形式で IPC コードのリストを返す想定なので、
# 必要に応じて「コードだけ抜き出して右ペインに表示する」など実装可。
# 例：
if "assistant" in [m["role"] for m in st.session_state.messages]:
    # 最後の assist メッセージを解析して IPC コードを箇条書きだけ抽出し、横に表示するなど
    last_ai = st.session_state.messages[-1]["content"]
    # 簡易的に「行頭が数字. 〜」を正規表現で拾うなどしてコードを抽出できる
    # この部分は後段の “拡張ポイント” で紹介
    # st.sidebar.markdown("### 提案された IPC コード")
    # st.sidebar.markdown(抜き出したリストを Markdown 化して表示)
    pass
