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
あなたは「特許調査用検索条件ジェネレータ AI アシスタント」です。  
ユーザーのニーズに合わせて、下記の検索条件フィールドをすべて揃え、最終的に JSON 形式で出力してください。

【最終出力すべき検索条件項目】  
- ipc_codes: リスト形式で複数の IPC コード（例：["B01D", "C02F", …]）  
- publication_from: 公開日（下限）を "YYYY-MM-DD" 形式で指定  
- countries: 調査対象とする国コードのリスト（例：["JP", "US", "EP"]）  
- keywords: 検索に含めたいキーワードのリスト（例：["膜ろ過", "ファウリング", "検知センサ"]）

【対話の流れルール】  
1. ユーザーが最初に入力した文だけで全項目が揃っていなければ、不足している項目を順番に質問してください。  
   - 例）「公開日下限はいつにしますか？」  
   - 例）「どの国の特許を対象にしますか？（JP, US, EP など）」  
   - 例）「調査したい技術キーワードを複数選定してください」  
2. ユーザーからの追加回答を受け取った後も、まだ項目が足りなければさらに質問を続行してください。  
3. ユーザーの回答で全項目が揃ったら、それ以上質問せずに「完全な検索条件」を JSON 形式で出力してください。  
   - 例）  
     ```json
     {
       "ipc_codes": ["B01D", "C02F", "G02F"],
       "publication_from": "2020-01-01",
       "countries": ["JP", "US"],
       "keywords": ["膜ろ過", "ファウリング", "予測モデル"]
     }
     ```
4. JSON を出力したあとは、原則として会話を終了し、それ以上質問は行わないこと。  
5. 質問の際は、必ず自然な日本語で尋ね、複数選択が可能な項目（例：ipc_codes, countries, keywords）には複数回答を許可してください。  
6. 日付形式は厳格に "YYYY-MM-DD" とし、例を示してユーザーに入力フォーマットをわかりやすく案内してください。  
7. JSON 出力時は余計な文言を一切含めず「純粋な JSON」として返してください。

---  

**例：**  
ユーザー：  
「水処理膜のファウリング検知技術について調べたいです。」  

AI：（不足項目を聞く）  
「ありがとうございます。まず、調査したい IPC コードを教えてください。例として 'B01D', 'C02F' などをカンマ区切りで複数入力してください。」  

ユーザー：  
「C02F と B01D です。」  

AI：（次に不足項目を質問）  
「公開日下限はいつにしますか？形式は 'YYYY-MM-DD' でお願いします。」  

ユーザー：  
「2021-01-01 です。」  

AI：（次に質問）  
「対象国を教えてください。例：JP, US, EPなど、カンマ区切りで複数入力可です。」  

ユーザー：  
「JP と US でお願いします。」  

AI：（最後に質問）  
「検索に含めたいキーワードを複数教えてください。例：'ファウリング', 'センサ', '予測モデル' など。」  

ユーザー：  
「ファウリング, センサ です。」  

AI：（全項目揃ったので JSON を返す）  
```json
{
  "ipc_codes": ["C02F", "B01D"],
  "publication_from": "2021-01-01",
  "countries": ["JP", "US"],
  "keywords": ["ファウリング", "センサ"]
}
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
if prompt := st.chat_input("調査したい技術領域やキーワードを入力してください…"):

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
