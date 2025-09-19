

import base64
import io
from PIL import Image
import json
from PIL import Image
import io
from typing import TypedDict, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse

from langgraph.graph import StateGraph, END
import base64
import json
from langfuse import get_client

from pydantic_data_models import GraphState, MedicalReport, Prescription, Bill
from graph_nodes import ocr_node, classify_node, extract_bill_node, extract_prescription_node, extract_report_node, route_documents


load_dotenv()

langfuse = get_client()
#create custom trace_id 
predefined_trace_id = Langfuse.create_trace_id()

if langfuse.auth_check():
    print("langfuse client is authenticated and ready")
else:
    print("authentication failed")
    

workflow = StateGraph(GraphState)

workflow.add_node("ocr", ocr_node)
workflow.add_node("classify", classify_node)
workflow.add_node("extract_bill", extract_bill_node)
workflow.add_node("extract_prescription", extract_prescription_node)
workflow.add_node("extract_report", extract_report_node)

workflow.set_entry_point("ocr")

workflow.add_edge("ocr", "classify")

workflow.add_conditional_edges(
    "classify",
    route_documents,
    {
        "extract_bill": "extract_bill",
        "extract_prescription": "extract_prescription",
        "extract_report": "extract_report",
        "end": END,
    },
)

workflow.add_edge("extract_bill", END)
workflow.add_edge("extract_prescription", END)
workflow.add_edge("extract_report", END)

langfuse_handler = CallbackHandler()

app = workflow.compile().with_config({"callbacks": [langfuse_handler]})
# app = workflow.compile()


# ----- USING LANGGRAPH SERVER 
# app = workflow.compile().with_config({"callbacks": [langfuse_handler]})
# def process_document(image_path: str):

#     print(f"\n--- Starting to process: {image_path} ---")
#     with Image.open(image_path) as img:
#         buffer = io.BytesIO()
#         if img.mode != 'RGB':
#             img = img.convert('RGB')
        
#         img.save(buffer, format="JPEG")
#         image_bytes = base64.b64encode(buffer.getvalue())

#     inputs = {"image_bytes": image_bytes}
#     result = app.invoke(input = inputs)
#     print(json.dumps(result.get("ocr_text"), indent=2))
#     print("\n--- Final Result ---")
#     print(json.dumps(result.get("extracted_json"), indent=2))
    
    
## adding Langfuse as callbackk to the invocation

def get_user_feedback_score(result: dict):
    return 1

def process_document(image_path: str):

    print(f"\n--- Starting to process: {image_path} ---")
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img.save(buffer, format="JPEG")
        image_bytes = base64.b64encode(buffer.getvalue())

    inputs = {"image_bytes": image_bytes}
    with langfuse.start_as_current_span(
         name="langgraph-request", trace_context={"trace_id": predefined_trace_id} ) as span:
    # LangGraph execution 
        result = app.invoke(input = inputs , config={"callbacks": [langfuse_handler]})
    # score using the span object
        span.score_trace(
            name="user-feedback",
            value=1,
            data_type="NUMERIC",
            comment="was helpful"
        )
        
    # score using current context
    #     langfuse.score_current_trace(
    #     name="user-feedback-opt-2",
    #     value= get_user_feedback_score(result),
    #     data_type="NUMERIC", # optional, langfuse will infer the type if not provided 
    #     comment="was helpful"
    #    )

    # result = app.invoke(inputs)
    print(json.dumps(result.get("ocr_text"), indent=2))
    print("\n--- Final Result ---")
    print(json.dumps(result.get("extracted_json"), indent=2))


process_document("OPD_report.png")