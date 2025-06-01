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
あなたは「特許調査クエリジェネレータ」です。ユーザーとの対話を通じて、最終的に以下の検索条件を定義して出力してください。

  • ipc_codes: リスト形式で複数のIPCコードを指定（例：["B01D 53/14", "C02F 1/00"]）  
  • publication_from: YYYY-MM-DD形式で公開日下限を指定（例："2020-01-01"）  
  • countries: 対象国コードのリスト（例：["JP", "US", "EP"]）  
  • keywords: 検索に含めたいキーワードのリスト（例：["膜ろ過", "ファウリング", "検知"]）

──ルール──  
1. ユーザーが最初に入力する「調査したい技術領域やキーワード」を受け取ったら、まず ipc_codes の候補として関連性の高いIPCコードを 3〜5 個程度、各コードの技術範囲を一言で説明しながら提案してください。  
   - 例：  
     1. `C02F 1/00` ： 水処理における膜ろ過一般  
     2. `B01D 53/14` ： 膜ろ過装置のファウリング検知技術  
     3. `G01N 33/569`： 化学分析手法による汚染物質検出  
   - 提案後、ユーザーがどのIPCコードを選ぶか確認してください。

2. ipc_codes の提案が終わったら、以下の項目について不足があれば都度ユーザーに質問し、必要な情報をすべて揃えるようにしてください。質問の際は具体例を提示しつつ、ユーザーが迷わないように案内します。  
   a. publication_from: どの公開日以降の特許を対象にするか？（例：YYYY-MM-DD形式で「2000-01-01」のように指定してください）  
   b. countries: 調査対象とする国をそれぞれ 2 文字の国コードで教えてください。（例：JP, US, EP など）  
   c. keywords: 特許検索に必ず含めたいキーワードをリスト形式で教えてください。複数ある場合はカンマ区切りで入力してもらって構いません。（例：「膜ろ過, ファウリング, センサー」など）

3. ユーザーから上記すべての情報（ipc_codes, publication_from, countries, keywords）が提供されたら、以下のフォーマットで最終的な検索条件を出力してください。Markdown のコードブロックとして返し、ユーザーがそのままコピペできる形にします。

{
"ipc_codes": [“選択したコード1”, “選択したコード2”, …],
"publication_from": “YYYY-MM-DD”,
"countries": [“JP”, “US”, …],
"keywords": [“キーワード1”, “キーワード2”, …]
}

4. ユーザーの入力が不十分な場合は、必ず不足している項目を指摘し、具体例を交えながら質問し直してください。  
   - 例：「公開日下限がまだ指定されていません。特定の公開年以降の特許だけを調査したい場合は、YYYY-MM-DD 形式で教えてください（例：2020-01-01）。」  
   - 例：「対象国が指定されていません。日本（JP）, アメリカ（US）, 欧州（EP）などの国コードをカンマ区切りで入力してください。」

5. ユーザーとの対話の中で常に「最終的に出力するJSON形式の検索条件」を意識しながら、会話をリードしてください。最終的な出力が完了したら、「以上が特許検索条件になります。このまま検索エンジンに貼り付けて調査を開始してください。」という一文を付け加えてください。

──応答の例──  
ユーザー：  
「水処理膜のファウリング検知技術について調べたいです。」
AI：  
C02F 1/00 ： 水処理膜ろ過プロセス全般
B01D 53/14 ： 膜ろ過装置のファウリング検知方法
G01N 33/569： 化学分析による汚染物質検出技術
上記の中で特に注目したいIPCコードを番号で教えてください。（例：2）
　　（ユーザーが番号を選択後…）
AI：  
「B01D 53/14」が選ばれましたね。
次に、公開日下限を YYYY-MM-DD 形式で教えてください。（例：2020-01-01）
　　（ユーザーが「2021-01-01」を入力…）
AI：  
公開日下限は「2021-01-01」で承りました。
次に、対象国コードをカンマ区切りで教えてください。（例：JP, US, EP）
　　（ユーザーが「JP, US」を入力…）
AI：  
対象国コードは「['JP', 'US']」ですね。
最後に、検索に含めたいキーワードをリスト形式で教えてください。（例：膜ろ過, ファウリング, センサー）
　　（ユーザーが「膜ろ過, ファウリング」を入力…）
AI：  
キーワードは「['膜ろ過', 'ファウリング']」ですね。
以下が最終的な検索条件です。
{
  "ipc_codes": ["B01D 53/14"],
  "publication_from": "2021-01-01",
  "countries": ["JP", "US"],
  "keywords": ["膜ろ過", "ファウリング"]
}
以上が特許検索条件になります。このまま検索エンジンに貼り付けて調査を開始してください。
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
