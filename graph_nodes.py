
import pprint
from dotenv import load_dotenv

from langfuse import get_client
from pydantic_data_models import MedicalReport, Prescription, Bill
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import TypedDict

load_dotenv()
load_dotenv()
langfuse = get_client()

vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.01)

modest_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.01)
mild_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.01)

class GraphState(TypedDict):
    image_bytes: bytes
    ocr_text: str
    document_type: str
    extracted_json: dict
    
    
def ocr_node(state: GraphState):
    
    print("--- Performing OCR ---")
    image_bytes = state["image_bytes"]
    
    prompt_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "You are an expert image extractor. Extract all text from this image accurately. Output a paragraph of text with no formatting. And also include a confidence score from 0.0 to 1.0 on the accuracy of the extracted text. Hand written true/false?",
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_bytes.decode()}",
            },
        ]
    )
    
    response = vision_model.invoke([prompt_message])
    print("--- OCR Complete ---")
    return {"ocr_text": response.content}

def classify_node(state: GraphState):
    """
    Classifies the document based on the OCR'd text.
    """
    print("--- Classifying Document ---")
    ocr_text = state["ocr_text"]
    prompt = langfuse.get_prompt("doc_classifier_prompt")
    prompt = prompt.compile(ocr_text = ocr_text)
    
    response = mild_model.invoke(prompt)
    doc_type = response.content.strip()
    print(f"--- Classified as: {doc_type} ---")
    return {"document_type": doc_type}

def extract_bill_node(state: GraphState):
    """
    Extracts information from a medical bill into a structured JSON.
    """
    print("--- Extracting Bill Details ---")
    ocr_text = state["ocr_text"]
    
    extractor_llm = modest_model.with_structured_output(Bill)
    
    prompt = f"""
    You are an expert in extracting information from medical documents.
    Based on the following text from a medical bill, extract the required information.

    Text:
    ---
    {ocr_text}
    ---
    """
    
    response = extractor_llm.invoke(prompt)
    print("--- Extraction Complete ---")
    return {"extracted_json": response.dict()}

def extract_prescription_node(state: GraphState):
    """
    Extracts information from a prescription into a structured JSON.
    """
    print("--- Extracting Prescription Details ---")
    ocr_text = state["ocr_text"]
    extractor_llm = modest_model.with_structured_output(Prescription)
    
    prompt = f"""
    You are an expert in extracting information from medical documents.
    Based on the following text from a prescription, extract the required information.

    Text:
    ---
    {ocr_text}
    ---
    """
    
    response = extractor_llm.invoke(prompt)
    print("--- Extraction Complete ---")
    return {"extracted_json": response.dict()}

def extract_report_node(state: GraphState):
    
    print("--- Extracting Report Details ---")
    ocr_text = state["ocr_text"]
    extractor_llm = modest_model.with_structured_output(MedicalReport)
    
    prompt = f"""
    You are an expert in extracting information from medical documents.
    Based on the following text from a medical report, extract the required information.

    Text:
    ---
    {ocr_text}
    ---
    """
    
    response = extractor_llm.invoke(prompt)
    print("--- Extraction Complete ---")
    return {"extracted_json": response.dict()}


def route_documents(state: GraphState):
    """
    Checks the document_type and decides which path to take.
    """
    doc_type = state["document_type"]
    print(f"--- Routing based on type: {doc_type} ---")
    if doc_type == "medical_bill":
        return "extract_bill"
    elif doc_type == "prescription":
        return "extract_prescription"
    elif doc_type == "medical_report":
        return "extract_report"
    else:
        return "end"