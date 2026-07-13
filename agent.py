#=======================================
#KEY, import
#=======================================

import os
from typing import TypedDict, List, Annotated, Optional, Any
import operator
import requests
import xml.etree.ElementTree as ET

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, InjectedState

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#=======================================
# GEMINI
#=======================================

# LLM모델 (결정성을 높이기 위해 temperature는 0으로 설정)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

#=======================================
# LANGGRAPH
#=======================================

# 에이전트들이 공유할 '기억 바구니(State)' 설계
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add] # 에이전트와 사용자의 대화 기록
    retriever: Optional[Any]        # 사용자별 pdf첨부 구분용
    user_intent: str      # 파악한 사용자의 의도 (예: "특허검색", "기업검색")
    search_query: str     # API에 던질 검색 키워드 (예: "스마트 가전 표준 특허")
    retrieved_data: str   # KIPRIS나 IPCAST에서 가져온 데이터 결과
    current_step: str     # 현재 진행 중인 노드 위치
    verification_failed: bool # 검증실예외처리

#=======================================
# External tool-calling
#=======================================

@tool
def search_kipris_api(query: str) -> str:
    """특허청 KIPRIS 데이터베이스에서 실시간으로 외부 특허 정보를 검색할 때 사용합니다.
    사용자가 최신 특허를 묻거나, 내부 문서(PDF)에 없는 외부 특허 정보를 찾아야 할 때 이 도구를 호출하세요.
    """
    api_key = os.getenv("KIPRIS_API_KEY", "YOUR_DEFAULT_KEY")
    
    # KIPRIS Plus 특허검색 실전 엔드포인트 URL
    url = "http://plus.kipris.or.kr/openapi/rest/PatentSvc/searchWord"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"accessKey": api_key, "searchWord": query, "numOfRows": 5}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            # XML 응답 데이터 파싱 및 텍스트 정제
            root = ET.fromstring(response.text)
            
            patent_list = []

            for item in root.findall(".//item"):
                title = item.findtext("inventionTitle", "명칭 없음")
                appl_no = item.findtext("applicationNumber", "번호 없음")
                applicant = item.findtext("applicantName", "출원인 없음")
                abstract = item.findtext("abstract", "요약 없음")
                
                patent_list.append(f"   - [특허명]: {title}\n     [출원번호]: {appl_no}\n     [출원인]: {applicant}\n     [요약]: {abstract}\n")
            
            if not patent_list:
                return f"KIPRIS 검색 결과 '{query}'에 대한 최신 특허 정보를 찾지 못했습니다."
                
            return "\n".join(patent_list)
        else:
            return f"KIPRIS API 연결 실패 (Status Code: {response.status_code})"
            
    except Exception as e:
        return f"KIPRIS API 호출 중 오류 발생: {str(e)}"

#=======================================
# RAG
#=======================================


pdf_retriever = None
global_retriever = None

def setup_pdf_database(pdf_path):
    
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250, length_function=len, is_separator_regex=False)
    splits = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore.as_retriever()
    
@tool
def search_internal_document(query: str, state: Annotated[dict, InjectedState]) -> str:
    """사용자가 업로드한 첨부 파일(PDF)에서 내용을 검색할 때 사용하는 도구입니다."""
    retriever = state.get("retriever")
    if retriever is None:
        return "[오류] 사용자가 아직 문서를 업로드하지 않았습니다."
    
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

#=======================================
# TOOLS
#=======================================

# node에 필요한 tools 정리
tools = [search_internal_document, search_kipris_api]
llm_with_tools = llm.bind_tools(tools)

#=======================================
# LANGGRAPH
#=======================================

def reasoning_node(state: AgentState):  
    print("[Reasoning Agent] 사용자의 의도를 분석하고 최적의 경로를 짠다...")
    messages = state["messages"]

    # 강력한 지시문 부여
    guard = ""
    if state.get("verification_failed"):
        guard = f"\n\n[가드레일] 직전 검색 결과가 신뢰 기준 미달({state.get('fail_reason')})이었습니다. 지어내지 말고 못 찾았다고 답하세요."
    sys_msg = SystemMessage(content= """
    너는 KEA의 수석 특허 분석 AI야.
    이전 대화 내역을 기억하고, 사용자의 질문에 자연스럽게 답해줘.
    질문에 '내부 문서'나 특허 관련 정보가 필요하면 반드시 'search_internal_document' 도구를 딱 1번만 호출해.
    검색된 데이터를 바탕으로 [기술 요약], [특허 번호], [핵심 청구항]을 마크다운으로 정리해줘.
    데이터가 부족하다면 "검색된 데이터에서 내용을 찾을 수 없습니다"라고 명확히 말해.
    어떠한 경우에도 빈칸("")만 출력하지 마.
    """ + guard)
    
    response = llm_with_tools.invoke([sys_msg] + messages)
    
    current_intent = "일반 대화"
    search_keyword = "없음"
    
    if response.tool_calls:
        tool_name = response.tool_calls[0]["name"]
        if tool_name == "search_internal_document":
            current_intent = "비정형 문서(PDF) RAG 검색"
        elif tool_name == "search_internal_excel_db":
            current_intent = "정형 데이터(Excel) 검색"
        else:
            current_intent = f"외부 API 호출 ({tool_name})"
            
        search_keyword = str(response.tool_calls[0]["args"])
    
    print(f"💡 [라우팅 결정] 의도: {current_intent} / 검색어: {search_keyword}")

    return {
        "messages": [response],
        "user_intent": current_intent,
        "search_query": search_keyword,
        "current_step": "의도 분류 및 라우팅 완료"
    }

def router_edge(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

def verification_node(state: AgentState):
    print("\n--- [VERIFICATION NODE] 가동: 데이터 신뢰성 검증 시작 ---")
    
    messages = state.get("messages", [])
    
    # 1. 가장 최근에 실행된 메시지가 도구(Tool)의 결과물인지 확인
    last_message = messages[-1] if messages else None
    
    verification_passed = True
    fail_reason = ""
    tool_output = ""
    
    if last_message and last_message.type == "tool":
        tool_output = str(last_message.content).strip()
        
        # [조건 1] 도구가 빈 값을 리턴했거나 검색 실패 메시지를 반환한 경우
        FAIL_MARKERS = ["[오류]", "찾지 못했습니다", "연결 실패", "호출 중 오류"]
        if not tool_output or any(m in tool_output for m in FAIL_MARKERS):
            fail_reason = "검색 결과 공백(Empty Result)"
        elif len(tool_output) < 20:
            fail_reason = "검색 텍스트 데이터 부족 (신뢰도 미달)"

    # 2. 검증 결과에 따른 분기 처리 및 감사 추적(Audit Trail) 기록
    if not verification_passed:
        print(f"⚠️ [검증 실패]: {fail_reason} -> LLM 환각 방지 조치 발동")
        return {"verification_failed": True, "fail_reason": fail_reason,
                "current_step": f"VERIFICATION_FAILED: {fail_reason}"}

tool_node = ToolNode(tools)

#=======================================
# LANGGRAPH 조합
#=======================================

# 그래프 조립 및 컴파일
graph_builder = StateGraph(AgentState)
graph_builder.add_node("reason", reasoning_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("verification", verification_node)

graph_builder.set_entry_point("reason")
graph_builder.add_conditional_edges("reason", router_edge)
graph_builder.add_edge("tools", "verification")
graph_builder.add_edge("verification", "reason")

tech_gpt = graph_builder.compile()

#=======================================
# run agent
#=======================================
def run_agent(chat_history: list, retriever=None) -> str:
    langchain_messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))

    initial_state = {"messages": langchain_messages, "retriever": retriever}
    final_output = None
    
    print(f"[LangGraph 추적 시작]")
    
    for output in tech_gpt.invoke(initial_state):
        node_name = list(output.keys())[0]
        node_update = output[node_name]

        if "messages" not in node_update or not node_update["messages"]:
            continue

        latest_msg = node_update["messages"][-1]
        print(f"방금 통과한 노드: [{node_name}]")

        if not hasattr(latest_msg, 'tool_calls') or not latest_msg.tool_calls:
            raw_content = latest_msg.content
            if isinstance(raw_content, list) and len(raw_content) > 0:
                final_output = raw_content[0].get('text', str(raw_content))
            else:
                final_output = str(raw_content)
        
    return final_output if final_output else "답변을 생성하지 못했습니다."

#=======================================
# additional features
#=======================================

def generate_automatic_report(chat_history):
    """채팅 기록을 바탕으로 마크다운 형식의 보고서를 생성합니다."""
    
    # 시스템 프롬프트 및 사용자와 AI의 대화 내역을 하나의 텍스트로 묶기
    conversation_text = ""
    for msg in chat_history:
        # 시스템 메시지나 초기 인사말은 제외하고 실제 대화만 추출
        if msg["role"] in ["user", "assistant"]:
            role_name = "사용자" if msg["role"] == "user" else "AI"
            conversation_text += f"[{role_name}]: {msg['content']}\n\n"

    # 보고서 생성을 위한 전용 프롬프트
    report_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 전문적인 기술/특허 분석 보고서 작성기입니다. 
        사용자와 AI가 나눈 아래의 대화 내역을 분석하여, 핵심 내용을 깔끔한 '마크다운(Markdown)' 형식의 보고서로 정리해주세요.
        
        [보고서 필수 구조]
        1. 요약 (Executive Summary)
        2. 주요 질의응답 내용 (Key Findings)
        3. 기술/특허 분석 인사이트 (Insights)
        4. 결론 및 향후 방향성 (Conclusion)
        
        말투는 "~함", "~임"과 같은 전문적인 개조식 문체를 사용하세요."""),
        ("user", "다음 대화 내역을 바탕으로 보고서를 작성해주세요:\n{conversation}")
    ])
    
    # 체인 구성 및 실행
    report_chain = report_prompt | llm 
    response = report_chain.invoke({"conversation": conversation_text})
    
    return response.content